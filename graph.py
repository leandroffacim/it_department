"""
graph.py
─────────────────────────────────────────────────────────────────────────────
Montagem do grafo LangGraph do IT Department.

Este módulo conecta todos os agentes num StateGraph compilado e pronto
para execução. É o ponto de entrada para qualquer código que queira
usar o departamento de TI.

Uso rápido:
    from graph import run_task

    result = run_task(
        task="Adicionar validação de entrada na função process_data()",
        repo_path="/home/user/meu_projeto",
    )

Uso avançado (streaming + human-in-the-loop):
    from graph import build_graph

    graph = build_graph(human_in_the_loop=True)
    config = {"configurable": {"thread_id": "sessao-01"}}

    for event in graph.stream(initial_state, config, stream_mode="values"):
        print(event)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from state import AgentState

# Importa set_base_path para sincronizar ALLOWED_BASE_PATH com repo_path em runtime
try:
    from tools.filesystem import set_base_path as _set_fs_base_path
except ImportError:
    _set_fs_base_path = None
from agents.supervisor import route_after_supervisor, supervisor_node

# ─────────────────────────────────────────────────────────────────────────────
# Import dos agentes (com fallback gracioso para os ainda não implementados)
# ─────────────────────────────────────────────────────────────────────────────

def _import_agent(module: str, func: str):
    """Importa um nó de agente; retorna stub se o módulo ainda não existir."""
    try:
        import importlib
        mod = importlib.import_module(f"agents.{module}")
        return getattr(mod, func)
    except (ImportError, AttributeError):
        def _stub(state: AgentState) -> AgentState:
            from agents.supervisor import record_agent_output
            msg = (
                f"⚠️ Agente '{module}' ainda não implementado.\n"
                f"Instrução recebida: {state.get('current_instruction', '')}\n\n"
                f"Retornando ao supervisor para replanejamento."
            )
            updates = record_agent_output(state, module, msg, status="warning")
            return {**state, **updates}
        _stub.__name__ = f"{func}_stub"
        return _stub


developer_node = _import_agent("developer", "developer_node")
qa_node        = _import_agent("qa",        "qa_node")
reviewer_node  = _import_agent("reviewer",  "reviewer_node")
devops_node    = _import_agent("devops",    "devops_node")
docs_node      = _import_agent("docs",      "docs_node")

# ─────────────────────────────────────────────────────────────────────────────
# Construção do grafo
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    human_in_the_loop: bool = False,
    interrupt_agents: Optional[list[str]] = None,
) -> "CompiledGraph":
    """
    Monta e compila o StateGraph do IT Department.

    Args:
        human_in_the_loop: Se True, pausa antes do nó 'developer' para
                           aprovação humana (padrão: False).
        interrupt_agents:  Lista customizada de agentes para pausar antes.
                           Sobrescreve human_in_the_loop se fornecida.

    Returns:
        Grafo compilado com checkpointer MemorySaver.

    Topologia:
                        ┌─────────────────────────────────────────┐
                        ↓                                         │
        START → supervisor → [developer|qa|reviewer|devops|docs] ─┘
                    │
                    └──→ END
    """
    builder = StateGraph(AgentState)

    # ── Registra nós ─────────────────────────────────────────────────────────
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("developer",  developer_node)
    builder.add_node("qa",         qa_node)
    builder.add_node("reviewer",   reviewer_node)
    builder.add_node("devops",     devops_node)
    builder.add_node("docs",       docs_node)

    # ── Ponto de entrada ──────────────────────────────────────────────────────
    builder.set_entry_point("supervisor")

    # ── Supervisor roteia para qualquer agente ou END ─────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "developer": "developer",
            "qa":        "qa",
            "reviewer":  "reviewer",
            "devops":    "devops",
            "docs":      "docs",
            END:          END,
        },
    )

    # ── Todos os agentes retornam ao supervisor ───────────────────────────────
    for agent in ["developer", "qa", "reviewer", "devops", "docs"]:
        builder.add_edge(agent, "supervisor")

    # ── Checkpointer (habilita persistência e human-in-the-loop) ─────────────
    memory = MemorySaver()

    # Define onde pausar para intervenção humana
    interrupts: list[str] = []
    if interrupt_agents is not None:
        interrupts = interrupt_agents
    elif human_in_the_loop:
        interrupts = ["developer"]  # pausa antes de qualquer escrita de código

    return builder.compile(
        checkpointer=memory,
        interrupt_before=interrupts if interrupts else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Estado inicial
# ─────────────────────────────────────────────────────────────────────────────

def create_initial_state(
    task: str,
    repo_path: str = ".",
) -> AgentState:
    """
    Cria o estado inicial para uma nova execução.

    Args:
        task:      Descrição do que deve ser feito.
        repo_path: Caminho do repositório local.

    Returns:
        AgentState pronto para ser passado ao graph.invoke() ou graph.stream().
    """
    resolved = str(Path(repo_path).resolve())

    # Sincroniza o base path das filesystem tools com o repo escolhido
    if _set_fs_base_path:
        _set_fs_base_path(resolved)

    return AgentState(
        task=task,
        repo_path=resolved,
        messages=[HumanMessage(content=task)],
        plan="",
        next_agent="",
        current_instruction="",
        iteration=0,
        routing_history=[],
        agent_outputs=[],
        artifacts={},
        human_approved=None,
        human_feedback=None,
        final_summary=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# API de alto nível
# ─────────────────────────────────────────────────────────────────────────────

def run_task(
    task: str,
    repo_path: str = ".",
    thread_id: str = "default",
    verbose: bool = True,
) -> AgentState:
    """
    Executa um task completo de forma síncrona.

    Args:
        task:      O que deve ser feito (ex: "adicionar testes para auth.py").
        repo_path: Caminho do repositório.
        thread_id: ID da thread para persistência (permite retomar depois).
        verbose:   Se True, imprime o progresso em tempo real.

    Returns:
        Estado final após a conclusão do task.

    Exemplo:
        state = run_task(
            task="Refatorar a função parse_config para usar dataclasses",
            repo_path="/home/user/meu_projeto",
        )
        print(state["final_summary"])
    """
    graph  = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state  = create_initial_state(task, repo_path)

    if verbose:
        _print_header(task, repo_path)

    final_state = None

    for event in graph.stream(state, config, stream_mode="values"):
        final_state = event

        if verbose:
            _print_event(event)

    if verbose:
        _print_footer(final_state)

    return final_state


def stream_task(
    task: str,
    repo_path: str = ".",
    thread_id: str = "default",
) -> Iterator[AgentState]:
    """
    Executa um task e retorna um gerador de estados (para UIs reativas).

    Exemplo:
        for state in stream_task("adicionar docstrings em utils.py"):
            ultimo_agente = state["routing_history"][-1] if state["routing_history"] else {}
            print(f"Agente: {ultimo_agente.get('agent', '?')}")
    """
    graph  = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state  = create_initial_state(task, repo_path)

    yield from graph.stream(state, config, stream_mode="values")


def resume_with_feedback(
    thread_id: str,
    approved: bool,
    feedback: str = "",
) -> AgentState:
    """
    Retoma uma execução pausada (human-in-the-loop).

    Use após o grafo ter pausado num interrupt_before.
    Se approved=False, o supervisor receberá o feedback e replanejará.

    Args:
        thread_id: ID da thread pausada.
        approved:  True para continuar, False para rejeitar e dar feedback.
        feedback:  Comentário/instrução do humano (usado se approved=False).

    Returns:
        Estado final após a retomada.

    Exemplo:
        # Graph pausou antes do developer
        final = resume_with_feedback(
            thread_id="sessao-01",
            approved=False,
            feedback="Não modifique auth.py, use apenas utils.py",
        )
    """
    graph  = build_graph(human_in_the_loop=True)
    config = {"configurable": {"thread_id": thread_id}}

    # Injeta a decisão humana no estado atual
    graph.update_state(
        config,
        {
            "human_approved": approved,
            "human_feedback": feedback if not approved else None,
        },
    )

    final_state = None
    for event in graph.stream(None, config, stream_mode="values"):
        final_state = event

    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de output no terminal
# ─────────────────────────────────────────────────────────────────────────────

AGENT_ICONS = {
    "supervisor": "🧠",
    "developer":  "👨‍💻",
    "qa":         "🧪",
    "reviewer":   "🔍",
    "devops":     "⚙️",
    "docs":       "📚",
}

def _print_header(task: str, repo_path: str) -> None:
    width = 60
    print("\n" + "═" * width)
    print("  🏢  IT DEPARTMENT  —  LangGraph Multi-Agent")
    print("═" * width)
    print(f"  📋 Task:  {task[:50]}{'...' if len(task) > 50 else ''}")
    print(f"  📂 Repo:  {repo_path}")
    print("─" * width)


def _print_event(state: AgentState) -> None:
    history = state.get("routing_history", [])
    if not history:
        return

    last = history[-1]
    agent    = last.get("agent", "?")
    reason   = last.get("reason", "")
    iteration = last.get("iteration", "?")

    if agent == "FINISH":
        return

    icon = AGENT_ICONS.get(agent, "🤖")
    print(f"\n  {icon}  [{iteration}] {agent.upper()}")
    if reason:
        print(f"      ↳ {reason[:70]}")


def _print_footer(state: AgentState | None) -> None:
    if not state:
        return

    width = 60
    iterations = state.get("iteration", 0)
    artifacts  = state.get("artifacts", {})
    files      = artifacts.get("files_changed", [])

    print("\n" + "─" * width)
    print("  ✅  CONCLUÍDO")
    print(f"  📊 Iterações:       {iterations}")
    print(f"  📝 Arquivos tocados: {len(files)}")
    if files:
        for f in files[:5]:
            print(f"      • {f}")
        if len(files) > 5:
            print(f"      ... e mais {len(files) - 5}")

    summary = state.get("final_summary")
    if summary:
        print(f"\n  📋 Resumo:\n  {summary[:200]}")

    print("═" * width + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Visualização do grafo (debug)
# ─────────────────────────────────────────────────────────────────────────────

def print_graph_structure() -> None:
    """Imprime a estrutura do grafo em ASCII (útil para debug)."""
    graph = build_graph()
    try:
        print(graph.get_graph().draw_ascii())
    except Exception:
        print("Grafo compilado com os nós:")
        print("  supervisor → [developer, qa, reviewer, devops, docs] → supervisor")
        print("  supervisor → END")


def save_graph_image(path: str = "graph.png") -> None:
    """Salva uma imagem PNG do grafo (requer graphviz instalado)."""
    graph = build_graph()
    try:
        img = graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(img)
        print(f"Grafo salvo em: {path}")
    except Exception as e:
        print(f"Não foi possível salvar imagem: {e}")
        print("Instale: pip install graphviz")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point direto
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--structure" in sys.argv:
        print_graph_structure()
        sys.exit(0)

    task      = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Analise o repositório e sugira melhorias"
    repo_path = os.environ.get("ITDEPT_REPO_PATH", ".")

    run_task(task=task, repo_path=repo_path, verbose=True)