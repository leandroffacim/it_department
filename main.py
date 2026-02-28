"""
main.py
─────────────────────────────────────────────────────────────────────────────
CLI do IT Department Multi-Agent System.

Uso:
    python main.py "sua tarefa aqui"
    python main.py "adicionar testes para auth.py" --repo ./meu_projeto
    python main.py --interactive
    python main.py --resume <thread-id> --approve
    python main.py --resume <thread-id> --feedback "não mexa em auth.py"

Comandos especiais no modo interativo:
    /status     → mostra estado atual da execução
    /history    → histórico de roteamentos
    /artifacts  → artefatos produzidos
    /tree       → árvore do repositório
    /exit       → encerra
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
import sys
import textwrap
import time
import uuid
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Carrega variáveis de ambiente do .env
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Verifica dependências antes de importar o resto
# ─────────────────────────────────────────────────────────────────────────────

def _check_dependencies() -> None:
    missing = []
    try:
        import langgraph       # noqa: F401
    except ImportError:
        missing.append("langgraph")
    try:
        import langchain_ollama  # noqa: F401
    except ImportError:
        missing.append("langchain-ollama")

    if missing:
        print("\n❌  Dependências ausentes:")
        for pkg in missing:
            print(f"    pip install {pkg}")
        print("\nInstale tudo de uma vez:")
        print("    pip install langgraph langchain-ollama langchain-core\n")
        sys.exit(1)

    # if not os.environ.get("ANTHROPIC_API_KEY"):
    #     print("\n❌  ANTHROPIC_API_KEY não definida.")
    #     print("    export ANTHROPIC_API_KEY='sua-chave-aqui'\n")
    #     sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Cores ANSI
# ─────────────────────────────────────────────────────────────────────────────

USE_COLOR = sys.stdout.isatty() and os.name != "nt"

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def cyan(t):    return _c(t, "96")
def green(t):   return _c(t, "92")
def yellow(t):  return _c(t, "93")
def red(t):     return _c(t, "91")
def bold(t):    return _c(t, "1")
def dim(t):     return _c(t, "2")
def purple(t):  return _c(t, "95")
def blue(t):    return _c(t, "94")

AGENT_COLORS = {
    "supervisor": purple,
    "developer":  green,
    "qa":         yellow,
    "reviewer":   cyan,
    "devops":     red,
    "docs":       blue,
}

AGENT_ICONS = {
    "supervisor": "🧠",
    "developer":  "👨‍💻",
    "qa":         "🧪",
    "reviewer":   "🔍",
    "devops":     "⚙️ ",
    "docs":       "📚",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de UI
# ─────────────────────────────────────────────────────────────────────────────

WIDTH = 70

def _line(char: str = "─", color=dim) -> None:
    print(color(char * WIDTH))

def _header() -> None:
    print()
    _line("═", bold)
    print(bold("  🏢  IT DEPARTMENT") + dim("  —  LangGraph Multi-Agent System"))
    _line("═", bold)

def _section(title: str, color=cyan) -> None:
    print()
    print(color(f"  ▸ {title}"))
    _line()

def _spinner(msg: str, duration: float = 0.0) -> None:
    """Mostra mensagem com indicador simples (sem threads)."""
    print(f"  {dim('...')} {msg}")

def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(
        textwrap.fill(line, width=WIDTH - indent, initial_indent=prefix,
                      subsequent_indent=prefix)
        if line.strip() else ""
        for line in text.splitlines()
    )

def _print_agent_event(routing_entry: dict) -> None:
    agent     = routing_entry.get("agent", "?")
    reason    = routing_entry.get("reason", "")
    iteration = routing_entry.get("iteration", "?")

    if agent == "FINISH":
        print(f"\n  {green('✅')} {bold('CONCLUÍDO')}  {dim(f'(iteração {iteration})')}")
        return

    color = AGENT_COLORS.get(agent, dim)
    icon  = AGENT_ICONS.get(agent, "🤖")
    label = color(bold(agent.upper()))

    print(f"\n  {icon} {label}  {dim(f'[{iteration}]')}")
    if reason:
        print(f"     {dim('↳')} {dim(reason[:65])}")

def _print_final_report(state: dict) -> None:
    if not state:
        return

    _line("═", bold)
    print(bold("  📋  RELATÓRIO FINAL"))
    _line()

    iterations    = state.get("iteration", 0)
    artifacts     = state.get("artifacts", {})
    files_changed = artifacts.get("files_changed", [])
    coverage      = artifacts.get("coverage_percent")
    tests_passed  = artifacts.get("tests_passed")
    tests_failed  = artifacts.get("tests_failed")
    verdict       = artifacts.get("review_verdict")
    commit_msg    = artifacts.get("commit_message")

    print(f"  {'Iterações:':20s} {iterations}")
    print(f"  {'Arquivos tocados:':20s} {len(files_changed)}")

    if files_changed:
        for f in files_changed[:6]:
            print(f"    {dim('•')} {f}")
        if len(files_changed) > 6:
            print(f"    {dim(f'... e mais {len(files_changed)-6}')}")

    if tests_passed is not None:
        status = green("PASSOU") if not tests_failed else red("FALHOU")
        print(f"  {'Testes:':20s} {status}  "
              f"{green(str(tests_passed))} passaram"
              + (f"  {red(str(tests_failed))} falharam" if tests_failed else ""))

    if coverage is not None:
        bar_filled = int(coverage / 5)
        bar = green("█" * bar_filled) + dim("░" * (20 - bar_filled))
        print(f"  {'Cobertura:':20s} {bar} {coverage}%")

    if verdict:
        colors = {"APROVADO": green, "PRECISA_AJUSTES": yellow,
                  "REPROVADO": red, "INCONCLUSIVO": dim}
        vc = colors.get(verdict, dim)
        print(f"  {'Code review:':20s} {vc(verdict)}")

    if commit_msg:
        print(f"  {'Commit:':20s} {dim(commit_msg[:50])}")

    # Histórico de roteamento compacto
    routing = state.get("routing_history", [])
    if routing:
        print(f"\n  {dim('Fluxo de execução:')}")
        flow = " → ".join(
            AGENT_COLORS.get(r["agent"], dim)(r["agent"])
            for r in routing
        )
        print(f"    {flow}")

    _line("═", bold)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Modo single task (não-interativo)
# ─────────────────────────────────────────────────────────────────────────────

def run_single(task: str, repo_path: str, thread_id: str) -> None:
    from graph import stream_task

    _header()
    print(f"  {bold('Task:')}    {task}")
    print(f"  {bold('Repo:')}    {repo_path}")
    print(f"  {bold('Thread:')}  {dim(thread_id)}")
    print(f"  {bold('Início:')}  {dim(datetime.now().strftime('%H:%M:%S'))}")
    _line()

    last_routing_len = 0
    final_state      = None
    start            = time.time()

    try:
        for state in stream_task(task, repo_path, thread_id):
            final_state = state

            # Mostra novos eventos de roteamento
            routing = state.get("routing_history", [])
            for entry in routing[last_routing_len:]:
                _print_agent_event(entry)
            last_routing_len = len(routing)

    except KeyboardInterrupt:
        print(f"\n\n  {yellow('⚠️  Interrompido pelo usuário.')}\n")
        if final_state:
            _print_final_report(final_state)
        sys.exit(0)
    except Exception as e:
        print(f"\n  {red(f'❌ Erro: {e}')}\n")
        raise

    elapsed = time.time() - start
    print(f"\n  {dim(f'Tempo total: {elapsed:.1f}s')}")
    _print_final_report(final_state)


# ─────────────────────────────────────────────────────────────────────────────
# Modo human-in-the-loop
# ─────────────────────────────────────────────────────────────────────────────

def run_with_hitl(task: str, repo_path: str, thread_id: str) -> None:
    from graph import build_graph, create_initial_state, resume_with_feedback

    _header()
    print(f"  {bold('Task:')}    {task}")
    print(f"  {bold('Repo:')}    {repo_path}")
    print(f"  {yellow('Modo:')}    Human-in-the-Loop (pausa antes do Developer)")
    _line()

    graph  = build_graph(human_in_the_loop=True)
    config = {"configurable": {"thread_id": thread_id}}
    state  = create_initial_state(task, repo_path)

    last_routing_len = 0
    final_state      = None

    # Fase 1: roda até o primeiro interrupt
    print(f"\n  {dim('Fase 1: planejamento...')}")
    for event in graph.stream(state, config, stream_mode="values"):
        final_state = event
        routing = event.get("routing_history", [])
        for entry in routing[last_routing_len:]:
            _print_agent_event(entry)
        last_routing_len = len(routing)

    if not final_state:
        print(red("  ❌ Nenhum estado retornado."))
        return

    # Pausa para revisão humana
    _section("⏸  REVISÃO HUMANA", yellow)
    plan = final_state.get("plan", "")
    if plan:
        print(f"\n  {bold('Plano do Supervisor:')}")
        print(_wrap(plan, indent=4))

    next_agent = final_state.get("next_agent", "?")
    instruction = final_state.get("current_instruction", "")
    print(f"\n  {bold('Próximo agente:')} {AGENT_COLORS.get(next_agent, dim)(next_agent.upper())}")
    if instruction:
        print(f"  {bold('Instrução:')}")
        print(_wrap(instruction[:300], indent=4))

    print()
    while True:
        try:
            choice = input(
                f"  {bold('Aprovar?')} "
                f"[{green('s')}]im  [{red('n')}]ão  [{yellow('f')}]eedback  "
                f"[{dim('q')}]uit : "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {yellow('Abortado.')}")
            sys.exit(0)

        if choice in ("s", "sim", "y", "yes", ""):
            approved  = True
            feedback  = ""
            break
        elif choice in ("n", "não", "nao", "no"):
            approved = False
            feedback = ""
            break
        elif choice in ("f", "feedback"):
            try:
                feedback = input("  Feedback: ").strip()
            except (EOFError, KeyboardInterrupt):
                feedback = ""
            approved = False
            break
        elif choice in ("q", "quit", "exit"):
            print(f"\n  {yellow('Saindo.')}")
            sys.exit(0)

    if approved:
        print(f"\n  {green('✅ Aprovado — continuando execução...')}")
    else:
        print(f"\n  {yellow('⚠️  Rejeitado')} — supervisor vai replanejar.")
        if feedback:
            print(f"  {dim('Feedback:')} {feedback}")

    # Fase 2: retoma após decisão humana
    final_state = resume_with_feedback(thread_id, approved, feedback)
    _print_final_report(final_state)


# ─────────────────────────────────────────────────────────────────────────────
# Retomar execução pausada
# ─────────────────────────────────────────────────────────────────────────────

def resume_execution(thread_id: str, approve: bool, feedback: str) -> None:
    from graph import resume_with_feedback

    _header()
    print(f"  {bold('Retomando thread:')} {thread_id}")
    action = green("APROVADO") if approve else yellow(f"REJEITADO — {feedback}")
    print(f"  {bold('Decisão:')} {action}")
    _line()

    final_state = resume_with_feedback(thread_id, approve, feedback)
    _print_final_report(final_state)


# ─────────────────────────────────────────────────────────────────────────────
# Modo interativo (REPL)
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive(repo_path: str) -> None:
    from graph import stream_task
    from tools.filesystem import get_repo_tree

    _header()
    print(f"  {bold('Repo:')} {repo_path}")
    print(f"  {dim('Digite sua tarefa ou um comando especial (/help para ajuda)')}")
    _line()

    session_id = uuid.uuid4().hex[:8]
    task_count = 0
    last_state: dict | None = None

    COMMANDS = {
        "/help":      "mostra esta ajuda",
        "/status":    "estado da última execução",
        "/history":   "histórico de roteamentos",
        "/artifacts": "artefatos produzidos",
        "/tree":      "árvore do repositório",
        "/repo":      "muda o repositório",
        "/exit":      "encerra",
    }

    def _show_help() -> None:
        print(f"\n  {bold('Comandos disponíveis:')}")
        for cmd, desc in COMMANDS.items():
            print(f"    {cyan(cmd):20s} {dim(desc)}")
        print()

    def _show_status() -> None:
        if not last_state:
            print(f"  {dim('Nenhuma execução ainda.')}\n")
            return
        print(f"\n  {bold('Iterações:')}  {last_state.get('iteration', 0)}")
        print(f"  {bold('Artefatos:')} {list(last_state.get('artifacts', {}).keys())}\n")

    def _show_history() -> None:
        if not last_state:
            print(f"  {dim('Nenhuma execução ainda.')}\n")
            return
        routing = last_state.get("routing_history", [])
        if not routing:
            print(f"  {dim('Sem histórico.')}\n")
            return
        print(f"\n  {bold('Histórico de roteamento:')}")
        for r in routing:
            color = AGENT_COLORS.get(r["agent"], dim)
            print(f"    {dim(str(r['iteration'])):>6}  {color(r['agent'].upper()):15s}  {dim(r.get('reason','')[:50])}")
        print()

    def _show_artifacts() -> None:
        if not last_state:
            print(f"  {dim('Nenhuma execução ainda.')}\n")
            return
        artifacts = last_state.get("artifacts", {})
        if not artifacts:
            print(f"  {dim('Sem artefatos.')}\n")
            return
        print(f"\n  {bold('Artefatos:')}")
        for k, v in artifacts.items():
            val = str(v)[:60] + ("..." if len(str(v)) > 60 else "")
            print(f"    {cyan(k):30s} {dim(val)}")
        print()

    def _show_tree() -> None:
        try:
            tree = get_repo_tree.invoke({"path": ".", "max_depth": 3})
            print()
            print(_wrap(tree, indent=2))
            print()
        except Exception as e:
            print(f"  {red(f'Erro: {e}')}\n")

    while True:
        try:
            prompt = f"\n  {cyan('it-dept')}:{dim(repo_path.split('/')[-1])}> "
            task = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {dim('Até logo! 👋')}\n")
            break

        if not task:
            continue

        # Comandos especiais
        if task == "/exit" or task == "/quit":
            print(f"\n  {dim('Até logo! 👋')}\n")
            break
        elif task == "/help":
            _show_help()
            continue
        elif task == "/status":
            _show_status()
            continue
        elif task == "/history":
            _show_history()
            continue
        elif task == "/artifacts":
            _show_artifacts()
            continue
        elif task == "/tree":
            _show_tree()
            continue
        elif task.startswith("/repo "):
            new_path = task[6:].strip()
            if Path(new_path).is_dir():
                repo_path = str(Path(new_path).resolve())
                print(f"  {green(f'Repositório: {repo_path}')}\n")
            else:
                print(f"  {red(f'Diretório não encontrado: {new_path}')}\n")
            continue
        elif task.startswith("/"):
            print(f"  {yellow(f'Comando desconhecido: {task}')}  (use /help)\n")
            continue

        # Executa task
        task_count += 1
        thread_id = f"{session_id}-{task_count}"

        print(f"  {dim(f'Thread: {thread_id}')}")
        _line()

        last_routing_len = 0
        start = time.time()

        try:
            for state in stream_task(task, repo_path, thread_id):
                last_state = state
                routing = state.get("routing_history", [])
                for entry in routing[last_routing_len:]:
                    _print_agent_event(entry)
                last_routing_len = len(routing)

        except KeyboardInterrupt:
            print(f"\n  {yellow('⚠️  Interrompido.')}")
            continue
        except Exception as e:
            print(f"\n  {red(f'❌ Erro: {e}')}")
            continue

        elapsed = time.time() - start
        print(f"\n  {dim(f'Concluído em {elapsed:.1f}s')}")
        _print_final_report(last_state)


# ─────────────────────────────────────────────────────────────────────────────
# Estrutura do projeto (diagnóstico)
# ─────────────────────────────────────────────────────────────────────────────

def show_project_status() -> None:
    """Mostra o status dos módulos do IT Department."""
    _header()
    print(f"  {bold('Status dos módulos:')}\n")

    base = Path(__file__).parent
    modules = [
        ("state.py",              "Estado compartilhado"),
        ("graph.py",              "Grafo principal"),
        ("agents/supervisor.py",  "Supervisor (IT Manager)"),
        ("agents/developer.py",   "Developer Agent"),
        ("agents/qa.py",          "QA Engineer Agent"),
        ("agents/reviewer.py",    "Code Reviewer Agent"),
        ("agents/devops.py",      "DevOps Agent"),
        ("agents/docs.py",        "Docs Writer Agent"),
        ("tools/filesystem.py",   "Filesystem Tools"),
        ("tools/git_tools.py",    "Git Tools"),
    ]

    for path, desc in modules:
        exists = (base / path).exists()
        icon   = green("✅") if exists else red("❌")
        print(f"  {icon}  {desc:30s} {dim(path)}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Argumentos CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="it-dept",
        description="IT Department — LangGraph Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Exemplos:
          python main.py "adicionar testes para auth.py"
          python main.py "criar Dockerfile" --repo ./meu_app
          python main.py "refatorar utils.py" --hitl
          python main.py --interactive --repo /home/user/projeto
          python main.py --resume abc123 --approve
          python main.py --resume abc123 --feedback "não mexa em auth.py"
          python main.py --status
        """),
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="Tarefa a executar (ex: 'adicionar testes para auth.py')",
    )
    parser.add_argument(
        "--repo", "-r",
        default=os.environ.get("ITDEPT_REPO_PATH", "."),
        help="Caminho do repositório (padrão: cwd ou $ITDEPT_REPO_PATH)",
    )
    parser.add_argument(
        "--thread", "-t",
        default=None,
        help="ID da thread para persistência (padrão: gerado automaticamente)",
    )
    parser.add_argument(
        "--hitl",
        action="store_true",
        help="Habilita Human-in-the-Loop (pausa antes do Developer)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Modo interativo (REPL)",
    )
    parser.add_argument(
        "--resume",
        metavar="THREAD_ID",
        help="Retoma uma execução pausada pelo thread-id",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Aprovação automática ao retomar (usado com --resume)",
    )
    parser.add_argument(
        "--feedback",
        default="",
        help="Feedback ao rejeitar (usado com --resume)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Mostra status dos módulos e sai",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Desabilita cores no output",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Output estruturado (JSON) para integração com outras ferramentas",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global USE_COLOR

    parser = _build_parser()
    args   = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    # Status dos módulos
    if args.status:
        show_project_status()
        return

    # Verifica dependências (exceto para --status)
    _check_dependencies()

    # Resolve repo_path
    repo_path = str(Path(args.repo).resolve())
    if not Path(repo_path).is_dir():
        print(f"\n{red(f'❌ Repositório não encontrado: {repo_path}')}\n")
        sys.exit(1)

    # Thread ID
    thread_id = args.thread or uuid.uuid4().hex[:12]

    # Modo: retomar execução pausada
    if args.resume:
        resume_execution(
            thread_id=args.resume,
            approve=args.approve,
            feedback=args.feedback,
        )
        return

    # Modo: interativo
    if args.interactive:
        run_interactive(repo_path)
        return

    # Modo: task única
    if not args.task:
        parser.print_help()
        print(f"\n{yellow('  ⚠️  Forneça uma task ou use --interactive')}\n")
        sys.exit(1)

    if args.hitl:
        run_with_hitl(args.task, repo_path, thread_id)
    else:
        run_single(args.task, repo_path, thread_id)


if __name__ == "__main__":
    main()
