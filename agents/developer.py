"""
developer.py
─────────────────────────────────────────────────────────────────────────────
Agente Developer do IT Department Multi-Agent System.

Responsabilidades:
  • Implementar novas features
  • Corrigir bugs
  • Refatorar código existente
  • Aplicar patches cirúrgicos (patch_file)
  • Fazer commits semânticos via git

Ferramentas disponíveis:
  filesystem → read_file, write_file, patch_file, list_directory, get_repo_tree
  git        → git_status, git_diff, git_add, git_commit, git_log

Padrão ReAct: o agente raciocina → age → observa em loop até concluir.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os

from llm_factory import make_llm
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from state import AgentState
from supervisor import record_agent_output
from tools.filesystem import DEVELOPER_TOOLS

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

DEVELOPER_MODEL = os.environ.get("ITDEPT_DEVELOPER_MODEL", "claude-sonnet-4-5")

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

DEVELOPER_SYSTEM_PROMPT = """\
Você é um desenvolvedor de software sênior especializado em Python.
Você trabalha em repositórios locais usando ferramentas de filesystem e git.

## SUAS FERRAMENTAS

- read_file       → leia SEMPRE antes de editar qualquer arquivo
- write_file      → cria ou sobrescreve um arquivo inteiro
- patch_file      → edição cirúrgica: substitui um trecho específico (preferível ao write_file para mudanças pequenas)
- list_directory  → veja a estrutura de um diretório
- get_repo_tree   → visão geral do repositório
- git_status      → veja arquivos modificados
- git_diff        → veja as mudanças antes de commitar
- git_add         → adicione arquivos ao staging
- git_commit      → commit com mensagem semântica

## PROCESSO DE TRABALHO

1. **Entenda antes de agir**
   - Use get_repo_tree e list_directory para entender a estrutura
   - Use read_file em TODOS os arquivos relevantes antes de qualquer edição
   - Verifique imports, dependências e convenções do projeto

2. **Implemente com qualidade**
   - Siga o estilo de código existente (indentação, naming, docstrings)
   - Prefira patch_file para mudanças pontuais (menos risco de regressão)
   - Use write_file apenas para arquivos novos ou rewrites completos
   - Adicione docstrings e type hints em funções novas

3. **Valide antes de commitar**
   - Use git_diff para revisar todas as mudanças
   - Confirme que nenhum arquivo irrelevante foi alterado

4. **Commit semântico**
   - Formato: `<tipo>(<escopo>): <descrição>`
   - Tipos: feat, fix, refactor, chore, test, docs
   - Ex: `feat(auth): add JWT token validation`
   - Faça git_add seguido de git_commit ao final

## BOAS PRÁTICAS

- Nunca altere arquivos fora do escopo da instrução recebida
- Se encontrar um bug não relacionado ao task, anote no report mas não corrija
- Prefira soluções simples e legíveis a soluções "inteligentes" e complexas
- Se a instrução for ambígua, opte pela interpretação mais conservadora

## REPORT FINAL

Ao concluir, escreva um report estruturado:
```
## O que foi feito
- <lista de mudanças>

## Arquivos modificados
- <caminho>: <descrição da mudança>

## Commit
<hash ou mensagem do commit>

## Observações
<bugs encontrados, dívidas técnicas, sugestões>
```
"""

# ─────────────────────────────────────────────────────────────────────────────
# Construção do agente ReAct
# ─────────────────────────────────────────────────────────────────────────────

def _build_developer_agent():
    """Constrói o agente ReAct com LLM + tools."""
    llm = make_llm("developer", temperature=0, max_tokens=4096)

    # Importa git tools aqui para não quebrar se git_tools.py ainda não existir
    try:
        from tools.git_tools import GIT_TOOLS
        tools = DEVELOPER_TOOLS + GIT_TOOLS
    except ImportError:
        tools = DEVELOPER_TOOLS

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SystemMessage(content=DEVELOPER_SYSTEM_PROMPT),
    )
    return agent


# Instância global (lazy init na primeira chamada)
_agent_instance = None


def _get_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = _build_developer_agent()
    return _agent_instance


# ─────────────────────────────────────────────────────────────────────────────
# Nó do grafo
# ─────────────────────────────────────────────────────────────────────────────

def developer_node(state: AgentState) -> AgentState:
    """
    Nó do Developer Agent no grafo LangGraph.

    Recebe a instrução do supervisor via state["current_instruction"],
    executa o ciclo ReAct (reason → act → observe) e registra o resultado.
    """
    instruction = state.get("current_instruction", "")
    repo_path   = state.get("repo_path", ".")
    task        = state.get("task", "")

    # Monta o prompt com contexto completo para o agente
    user_prompt = f"""\
## TASK ORIGINAL
{task}

## SUA INSTRUÇÃO (do IT Manager)
{instruction}

## REPOSITÓRIO
{repo_path}

Execute a instrução acima. Comece explorando o repositório se necessário,
implemente as mudanças, valide com git_diff e faça o commit ao final.
"""

    try:
        agent = _get_agent()

        # Invoca o agente ReAct — ele faz o loop tool_calls internamente
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_prompt}]
        })

        # Extrai o texto final da resposta
        final_message = result["messages"][-1]
        output = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )

        # Determina status baseado em keywords no output
        status = _infer_status(output)

        # Extrai artefatos do output
        artifacts = _extract_artifacts(output)

    except Exception as e:
        output = f"❌ Erro no Developer Agent: {type(e).__name__}: {e}"
        status = "error"
        artifacts = {}

    # Registra no estado e retorna
    updates = record_agent_output(
        state=state,
        agent_name="developer",
        output=output,
        status=status,
        artifacts=artifacts,
    )

    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_status(output: str) -> str:
    """Infere o status da execução com base no conteúdo do output."""
    output_lower = output.lower()

    error_keywords = [
        "erro", "error", "falha", "failed", "exception",
        "traceback", "não foi possível", "could not",
    ]
    warning_keywords = [
        "aviso", "warning", "atenção", "cuidado",
        "parcialmente", "incompleto",
    ]

    if any(kw in output_lower for kw in error_keywords):
        # Erros de ferramenta no meio do processo são normais (ReAct recupera)
        # Só marca como erro se aparecer no report final
        if output_lower.count("erro") > 2 or "❌" in output:
            return "error"

    if any(kw in output_lower for kw in warning_keywords):
        return "warning"

    return "success"


def _extract_artifacts(output: str) -> dict:
    """
    Extrai informações estruturadas do report do agente.
    Retorna um dict com artefatos identificados no texto.
    """
    import re
    artifacts = {}

    # Arquivos modificados
    files_pattern = re.findall(r'`([^`]+\.(?:py|js|ts|json|yaml|yml|toml|md|txt))`', output)
    if files_pattern:
        artifacts["files_changed"] = list(set(files_pattern))

    # Mensagem de commit
    commit_pattern = re.search(
        r'(?:feat|fix|refactor|chore|test|docs)\([^)]+\):[^\n]+',
        output,
    )
    if commit_pattern:
        artifacts["commit_message"] = commit_pattern.group().strip()

    # Marca que código foi alterado (QA precisa rodar depois)
    if any(kw in output.lower() for kw in ["write_file", "patch_file", "commit"]):
        artifacts["code_changed"] = True

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Exportações
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["developer_node"]