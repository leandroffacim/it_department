"""
supervisor.py
─────────────────────────────────────────────────────────────────────────────
O coração do IT Department Multi-Agent System.

O Supervisor age como um IT Manager experiente:
  1. Interpreta o request do usuário
  2. Analisa o estado atual do repositório
  3. Cria um plano de execução
  4. Roteia para o agente correto a cada passo
  5. Avalia o resultado e decide: continuar, iterar ou finalizar

Padrão: Supervisor Pattern (LangGraph docs)
  user → supervisor → agent_X → supervisor → agent_Y → ... → END

Roteamento dinâmico via LLM — o supervisor decide quem trabalha
baseado no plano, no estado atual e nos outputs anteriores dos agentes.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm_factory import make_llm
from langgraph.graph import END

import logging

# Silencia logs verbosos do LangChain/LangSmith que poluem o terminal
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("langchain_anthropic").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.ERROR)

logger = logging.getLogger("it_department.supervisor")

from state import AgentState

# ─────────────────────────────────────────────────────────────────────────────
# Configuração do LLM
# ─────────────────────────────────────────────────────────────────────────────

# Modelo usado pelo supervisor — pode ser sobrescrito via env
SUPERVISOR_MODEL = os.environ.get("ITDEPT_SUPERVISOR_MODEL", "claude-opus-4-5")

# Máximo de iterações antes de forçar o encerramento (evita loops infinitos)
MAX_ITERATIONS = int(os.environ.get("ITDEPT_MAX_ITERATIONS", "12"))

# Agentes disponíveis para delegação
AVAILABLE_AGENTS = Literal["developer", "qa", "reviewer", "devops", "docs", "FINISH"]

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt do Supervisor
# ─────────────────────────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM_PROMPT = """\
Você é o IT Manager de um departamento de tecnologia altamente eficiente.
Você coordena uma equipe de agentes especializados para desenvolver, manter
e melhorar repositórios de código.

## SUA EQUIPE

| Agente      | Especialidade                                              |
|-------------|------------------------------------------------------------|
| developer   | Escreve, refatora e implementa código. Faz commits git.    |
| qa          | Roda testes, linting, type checking. Gera relatórios.      |
| reviewer    | Code review, segurança, métricas de qualidade, anti-patterns. |
| devops      | Dependências, Docker, CI/CD, configs de ambiente.          |
| docs        | README, docstrings, changelog, diagramas de arquitetura.   |

## SEU PROCESSO DE DECISÃO

Para cada turno, você recebe o estado atual e deve responder com um JSON.
O campo "next_agent" deve conter EXATAMENTE um dos valores abaixo — nada mais:
  "developer"  "qa"  "reviewer"  "devops"  "docs"  "FINISH"

Exemplo de resposta válida:
```json
{
  "thinking": "o developer criou o arquivo, preciso rodar QA agora",
  "next_agent": "qa",
  "instruction": "rode pytest e ruff no arquivo src/auth.py recém criado",
  "reason": "código novo sempre precisa passar pelo QA antes de finalizar",
  "plan_update": null
}
```

## REGRAS DE ROTEAMENTO

1. **Novo código** → developer → qa → reviewer → (docs se necessário) → FINISH
2. **Bug fix**     → developer → qa → reviewer → FINISH  
3. **Refatoração** → reviewer (análise) → developer → qa → reviewer (validação) → FINISH
4. **Docs only**   → docs → FINISH
5. **Setup/infra** → devops → qa → FINISH
6. **Análise only**→ reviewer → FINISH

## QUANDO FINALIZAR (FINISH)

- Todos os requisitos do task estão atendidos
- QA passou (testes + linting) após qualquer mudança de código
- Code review aprovado
- Nenhum agente retornou erros não resolvidos
- Iterações chegaram no limite máximo (force finish com nota)

## QUALIDADE

- Nunca finalize se QA ainda não rodou após uma mudança de código
- Se um agente falhar 2x na mesma tarefa, tente uma abordagem diferente
- Prefira passos menores e iterativos a grandes mudanças de uma vez
- Documente o raciocínio no campo "thinking"

## FORMATO DO OUTPUT

Responda SOMENTE com o JSON acima. Sem markdown, sem texto extra.
O campo "instruction" deve ser claro o suficiente para o agente executar
sem precisar de contexto adicional — inclua nomes de arquivo, funções, etc.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt de planejamento inicial
# ─────────────────────────────────────────────────────────────────────────────

PLANNING_PROMPT = """\
Você recebeu um novo task. Crie o plano de execução inicial.

TASK: {task}
REPO: {repo_path}
CONTEXTO DO REPO:
{repo_context}

Responda com JSON:
```json
{{
  "plan": "plano detalhado em etapas numeradas",
  "estimated_steps": <número estimado de turnos>,
  "first_agent": "developer",
  "first_instruction": "instrução detalhada para o primeiro agente",
  "complexity": "low | medium | high",
  "thinking": "sua análise do task e por onde começar"
}}
```

Responda SOMENTE com o JSON.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt de roteamento (chamado a cada iteração)
# ─────────────────────────────────────────────────────────────────────────────

ROUTING_PROMPT = """\
## ESTADO ATUAL

Task original: {task}
Plano: {plan}
Iteração: {iteration}/{max_iterations}
Timestamp: {timestamp}

## OUTPUTS DOS AGENTES (mais recentes primeiro)

{agent_outputs}

## HISTÓRICO DE ROTEAMENTO

{routing_history}

## ARTEFATOS GERADOS

{artifacts_summary}

Analise o estado e decida o próximo passo. Lembre-se:
- Se houve mudança de código, QA deve rodar antes de FINISH
- Se iteração >= {max_iterations}, force FINISH com nota sobre o limite

Responda SOMENTE com o JSON de decisão.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_AGENTS = {"developer", "qa", "reviewer", "devops", "docs", "FINISH"}


def _parse_json_response(raw: str) -> dict:
    """
    Extrai e parseia JSON da resposta do LLM.
    Resiliente a: markdown code blocks, texto antes/depois, múltiplos blocos.
    Após parsear, valida e sanitiza o campo next_agent.
    """
    import re

    clean = raw.strip()

    # 1. Tenta extrair bloco ```json ... ```
    m = re.search(r'```json\s*(.*?)\s*```', clean, re.DOTALL)
    if m:
        clean = m.group(1).strip()
    else:
        # 2. Tenta extrair bloco ``` ... ```
        m = re.search(r'```\s*(.*?)\s*```', clean, re.DOTALL)
        if m:
            clean = m.group(1).strip()
        else:
            # 3. Tenta pegar o primeiro objeto JSON completo { ... }
            m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', clean, re.DOTALL)
            if m:
                clean = m.group(0).strip()

    # Tenta parsear
    data: dict | None = None
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Último recurso: regex greedy para o maior bloco JSON
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                pass

    if data is None:
        raise ValueError(f"Não foi possível parsear JSON da resposta:\n{raw[:500]}")

    # ── Sanitiza next_agent ────────────────────────────────────────────────
    # O LLM às vezes retorna o exemplo literal do prompt ou valores inválidos
    agent = str(data.get("next_agent", "")).strip().strip('"').strip("'")

    # Remove pipe e pega só o primeiro token (ex: "developer | qa" → "developer")
    if "|" in agent:
        agent = agent.split("|")[0].strip()

    # Remove prefixos comuns que o LLM adiciona
    for prefix in ("agente:", "agent:", "próximo:", "next:"):
        if agent.lower().startswith(prefix):
            agent = agent[len(prefix):].strip()

    # Normaliza capitalização para FINISH
    if agent.lower() == "finish":
        agent = "FINISH"

    if agent not in VALID_AGENTS:
        # Tenta match parcial (ex: "developer_node" → "developer")
        matched = next((v for v in VALID_AGENTS if v in agent.lower()), None)
        agent = matched if matched else "FINISH"

    data["next_agent"] = agent
    return data


def _format_agent_outputs(state: AgentState) -> str:
    """Formata os outputs dos agentes para o prompt de roteamento."""
    outputs = state.get("agent_outputs", [])
    if not outputs:
        return "  (nenhum output ainda)"

    lines = []
    for entry in reversed(outputs[-6:]):  # últimos 6 outputs
        agent = entry.get("agent", "?")
        output = entry.get("output", "")
        ts = entry.get("timestamp", "")
        status = entry.get("status", "")
        icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
        lines.append(f"{icon} [{agent.upper()}] {ts}\n{output[:800]}\n")

    return "\n".join(lines)


def _format_artifacts(state: AgentState) -> str:
    """Formata o resumo dos artefatos para o prompt."""
    artifacts = state.get("artifacts", {})
    if not artifacts:
        return "  (nenhum artefato)"
    return "\n".join(f"  • {k}: {str(v)[:100]}" for k, v in artifacts.items())


def _format_routing_history(state: AgentState) -> str:
    """Formata o histórico de roteamentos."""
    history = state.get("routing_history", [])
    if not history:
        return "  (início)"
    return "\n".join(
        f"  {i+1}. {h['agent'].upper()} — {h['reason'][:80]}"
        for i, h in enumerate(history[-8:])
    )


def _get_repo_context(state: AgentState) -> str:
    """
    Gera um resumo rápido do repositório para o planejamento.
    Usa get_repo_tree se disponível.
    """
    repo_path = state.get("repo_path", ".")
    try:
        from tools.filesystem import get_repo_tree, list_directory
        tree = get_repo_tree.invoke({"path": repo_path, "max_depth": 2})
        return tree
    except Exception:
        return f"Repositório em: {repo_path} (tree não disponível)"


# ─────────────────────────────────────────────────────────────────────────────
# Nó principal: supervisor_node
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> AgentState:
    """
    Nó do Supervisor no grafo LangGraph.

    Responsabilidades:
    - Na primeira iteração: cria o plano de execução
    - Nas demais: avalia outputs e decide o próximo agente
    - Atualiza o estado com: next_agent, current_instruction, plan, routing_history
    """
    llm = make_llm("supervisor", temperature=0, max_tokens=1024)

    iteration  = state.get("iteration", 0)
    plan       = state.get("plan", "")
    task       = state.get("task", "")

    # ── Fase 1: Planejamento inicial ─────────────────────────────────────────
    if iteration == 0 or not plan:
        repo_context = _get_repo_context(state)

        planning_prompt = PLANNING_PROMPT.format(
            task=task,
            repo_path=state.get("repo_path", "."),
            repo_context=repo_context,
        )

        response = llm.invoke([
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=planning_prompt),
        ])

        try:
            data = _parse_json_response(response.content)
        except ValueError as e:
            # Se o LLM não retornar JSON válido, usa fallback seguro
            data = {
                "plan": f"Executar task: {task}",
                "first_agent": "developer",
                "first_instruction": task,
                "complexity": "medium",
                "thinking": str(e),
                "estimated_steps": 3,
            }

        # Sanitiza first_agent com a mesma lógica do next_agent
        first_agent = str(data.get("first_agent", "developer")).strip()
        if "|" in first_agent:
            first_agent = first_agent.split("|")[0].strip()
        if first_agent.lower() == "finish":
            first_agent = "developer"  # não faz sentido começar com FINISH
        if first_agent not in VALID_AGENTS - {"FINISH"}:
            first_agent = "developer"
        data["first_agent"] = first_agent

        logger.debug("Planning response parsed: agent=%s complexity=%s",
                     first_agent, data.get("complexity"))

        new_message = AIMessage(
            content=(
                f"📋 **Plano criado** (complexidade: {data.get('complexity', '?')})\n\n"
                f"{data.get('plan', '')}\n\n"
                f"🚀 Começando com: **{data.get('first_agent', '?').upper()}**\n"
                f"_{data.get('thinking', '')}_"
            )
        )

        return {
            **state,
            "plan":                data.get("plan", ""),
            "next_agent":          data.get("first_agent", "developer"),
            "current_instruction": data.get("first_instruction", task),
            "iteration":           1,
            "routing_history":     [],
            "agent_outputs":       state.get("agent_outputs", []),
            "artifacts":           state.get("artifacts", {}),
            "messages":            state["messages"] + [new_message],
        }

    # ── Fase 2: Roteamento iterativo ─────────────────────────────────────────
    routing_prompt = ROUTING_PROMPT.format(
        task=task,
        plan=plan,
        iteration=iteration,
        max_iterations=MAX_ITERATIONS,
        timestamp=datetime.now().strftime("%H:%M:%S"),
        agent_outputs=_format_agent_outputs(state),
        routing_history=_format_routing_history(state),
        artifacts_summary=_format_artifacts(state),
    )

    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=routing_prompt),
    ])

    try:
        data = _parse_json_response(response.content)
    except ValueError:
        # Fallback: se não conseguiu parsear, encerra com aviso
        data = {
            "next_agent":   "FINISH",
            "instruction":  "",
            "reason":       "Erro ao parsear resposta do supervisor",
            "thinking":     "Encerrando por segurança",
            "plan_update":  None,
        }

    next_agent  = data.get("next_agent", "FINISH")
    instruction = data.get("instruction", "")
    reason      = data.get("reason", "")
    thinking    = data.get("thinking", "")
    plan_update = data.get("plan_update")

    # Guarda histórico de roteamento
    routing_history = state.get("routing_history", []) + [{
        "iteration": iteration,
        "agent":     next_agent,
        "reason":    reason,
        "timestamp": datetime.now().isoformat(),
    }]

    # Mensagem visível no chat
    if next_agent == "FINISH":
        icon = "🏁"
        msg_content = (
            f"{icon} **Task concluída!** (iteração {iteration})\n\n"
            f"_{thinking}_"
        )
    else:
        icons = {
            "developer": "👨‍💻", "qa": "🧪", "reviewer": "🔍",
            "devops": "⚙️", "docs": "📚",
        }
        icon = icons.get(next_agent, "🤖")
        msg_content = (
            f"{icon} **→ {next_agent.upper()}** (iteração {iteration})\n\n"
            f"**Motivo:** {reason}\n"
            f"**Instrução:** {instruction}\n\n"
            f"_{thinking}_"
        )

    new_message = AIMessage(content=msg_content)

    updated_plan = plan_update if plan_update else plan

    return {
        **state,
        "next_agent":          next_agent,
        "current_instruction": instruction,
        "plan":                updated_plan,
        "iteration":           iteration + 1,
        "routing_history":     routing_history,
        "messages":            state["messages"] + [new_message],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Função de roteamento condicional (usada no grafo)
# ─────────────────────────────────────────────────────────────────────────────

def route_after_supervisor(state: AgentState) -> str:
    """
    Lê state["next_agent"] e retorna o nome do próximo nó.
    Chamada como conditional_edge a partir do nó supervisor.

    Retorna:
        Nome do nó destino ou END do LangGraph.
    """
    next_agent = state.get("next_agent", "FINISH")
    iteration  = state.get("iteration", 0)

    # Guarda-chuva contra loops infinitos
    if iteration > MAX_ITERATIONS:
        return END

    if next_agent == "FINISH":
        return END

    valid_agents = {"developer", "qa", "reviewer", "devops", "docs"}
    if next_agent not in valid_agents:
        # Agente desconhecido — encerra com segurança
        return END

    return next_agent


# ─────────────────────────────────────────────────────────────────────────────
# Helper para registrar output de um agente no estado
# (cada agente chama isso ao terminar seu trabalho)
# ─────────────────────────────────────────────────────────────────────────────

def record_agent_output(
    state: AgentState,
    agent_name: str,
    output: str,
    status: str = "success",
    artifacts: dict | None = None,
) -> dict:
    """
    Utilitário que cada agente usa para registrar seu resultado no estado.

    Args:
        state:      Estado atual do grafo.
        agent_name: Nome do agente ("developer", "qa", etc.).
        output:     Texto do resultado.
        status:     "success" | "warning" | "error"
        artifacts:  Dicionário de artefatos produzidos (opcional).

    Returns:
        Dict com as atualizações de estado para retornar no nó.

    Exemplo de uso em developer.py:
        return record_agent_output(state, "developer", resultado, "success",
                                   artifacts={"files_changed": ["src/main.py"]})
    """
    entry = {
        "agent":     agent_name,
        "output":    output,
        "status":    status,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "iteration": state.get("iteration", 0),
    }

    updated_outputs = state.get("agent_outputs", []) + [entry]

    updated_artifacts = {**state.get("artifacts", {})}
    if artifacts:
        updated_artifacts.update(artifacts)

    # Mensagem visível no histórico
    status_icon = {"success": "✅", "warning": "⚠️", "error": "❌"}.get(status, "ℹ️")
    message = HumanMessage(
        content=(
            f"{status_icon} **{agent_name.upper()} report:**\n\n{output}"
        ),
        name=agent_name,
    )

    return {
        "agent_outputs": updated_outputs,
        "artifacts":     updated_artifacts,
        "messages":      state["messages"] + [message],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Exportações públicas
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "supervisor_node",
    "route_after_supervisor",
    "record_agent_output",
    "MAX_ITERATIONS",
    "AVAILABLE_AGENTS",
]