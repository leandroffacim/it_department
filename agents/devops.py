"""
devops.py
─────────────────────────────────────────────────────────────────────────────
Agente DevOps do IT Department Multi-Agent System.

Responsabilidades:
  • Gerenciar dependências (pip, poetry, npm)
  • Criar/atualizar Dockerfile e docker-compose
  • Configurar variáveis de ambiente (.env)
  • Setup de CI/CD (GitHub Actions)
  • Verificar saúde do ambiente (versões, ports, processos)
  • Criar scripts de setup e Makefile

Ferramentas:
  shell_tools → run_pip, run_docker, check_environment, run_make
  fs_tools    → read_file, write_file, patch_file, list_directory, create_directory
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from llm_factory import make_llm
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from state import AgentState
from supervisor import record_agent_output
from tools.filesystem import DEVOPS_TOOLS

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

DEVOPS_MODEL   = os.environ.get("ITDEPT_DEVOPS_MODEL", "claude-sonnet-4-5")
DEVOPS_TIMEOUT = int(os.environ.get("ITDEPT_DEVOPS_TIMEOUT", "120"))

try:
    from tools.filesystem import ALLOWED_BASE_PATH
except ImportError:
    ALLOWED_BASE_PATH = Path(os.environ.get("ITDEPT_BASE_PATH", str(Path.cwd()))).resolve()

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

DEVOPS_SYSTEM_PROMPT = """\
Você é um DevOps Engineer sênior especializado em Python e infraestrutura moderna.
Sua missão é garantir que o projeto roda de forma confiável em qualquer ambiente.

## SUAS FERRAMENTAS

- check_environment   → verifica versões, dependências instaladas, variáveis de ambiente
- run_pip             → instala, remove ou lista pacotes Python
- check_dependencies  → analisa requirements e detecta conflitos/desatualizações
- run_docker          → comandos docker (build, run, compose)
- scan_ports          → verifica portas em uso no sistema
- run_make            → executa targets de um Makefile
- read_file           → leia configs existentes antes de editar
- write_file          → crie novos arquivos de configuração
- patch_file          → edite configurações existentes de forma cirúrgica
- create_directory    → crie estrutura de pastas
- list_directory      → explore a estrutura do projeto
- move_file           → mova ou renomeie arquivos de config

## PROCESSO DE TRABALHO

1. **Audite o ambiente atual**
   - check_environment para ver o que está instalado
   - Leia requirements.txt, pyproject.toml, Dockerfile se existirem
   - Verifique se há .env ou .env.example

2. **Execute a instrução**
   - Gerenciamento de deps: sempre gere requirements.txt atualizado
   - Docker: valide sintaxe, use multi-stage builds quando possível
   - CI/CD: workflows em .github/workflows/
   - Env vars: NUNCA commite segredos reais — use .env.example com placeholders

3. **Documente as mudanças**
   - Atualize README se adicionou novos requisitos de setup
   - Mantenha .env.example sincronizado com .env

## BOAS PRÁTICAS

- Sempre use versões pinadas em requirements.txt (pacote==versão)
- Dockerfile: use imagens Alpine ou Slim quando possível
- Separe dependências de dev e produção
- Use multi-stage builds para imagens menores
- GitHub Actions: cache de dependências para builds rápidos
- Makefile: targets padrão (install, test, lint, run, docker-build)

## TEMPLATES QUE VOCÊ CONHECE

Você tem templates mentais prontos para:
  • Dockerfile Python (FastAPI, Django, Flask, script simples)
  • docker-compose.yml (app + postgres + redis)
  • .github/workflows/ci.yml (test + lint + type-check)
  • Makefile com targets padrão
  • pyproject.toml com ruff + mypy + pytest configurados
  • .env.example bem documentado

## REPORT FINAL

```
## O que foi feito
- <lista de mudanças>

## Arquivos criados/modificados
- <caminho>: <descrição>

## Ações necessárias do desenvolvedor
- <comandos para rodar após esta mudança>

## Avisos
- <possíveis problemas ou trade-offs>
```
"""

# ─────────────────────────────────────────────────────────────────────────────
# Shell Tools do DevOps
# ─────────────────────────────────────────────────────────────────────────────

def _run_cmd(
    args: list[str],
    cwd: Optional[str] = None,
    timeout: int = DEVOPS_TIMEOUT,
    env_extra: Optional[dict] = None,
) -> tuple[str, str, int]:
    """Executa um comando e retorna (stdout, stderr, returncode)."""
    workdir = cwd or str(ALLOWED_BASE_PATH)
    env = {**os.environ, **(env_extra or {})}
    try:
        result = subprocess.run(
            args,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError as e:
        return "", f"Comando não encontrado: {e}", 127
    except subprocess.TimeoutExpired:
        return "", f"Timeout após {timeout}s.", 1
    except Exception as e:
        return "", f"Erro inesperado: {e}", 1


@tool
def check_environment(repo_path: Optional[str] = None) -> str:
    """
    Audita o ambiente de desenvolvimento: Python, pip, ferramentas instaladas,
    variáveis de ambiente relevantes e arquivos de configuração presentes.

    Args:
        repo_path: Diretório raiz do repositório.

    Returns:
        Relatório completo do ambiente atual.
    """
    cwd = repo_path or str(ALLOWED_BASE_PATH)
    sections: list[str] = ["🔍 Auditoria do Ambiente\n" + "─" * 50]

    # Python e pip
    for cmd, label in [
        (["python", "--version"],  "Python"),
        (["python", "-m", "pip", "--version"], "pip"),
        (["git", "--version"],     "Git"),
        (["docker", "--version"],  "Docker"),
        (["docker", "compose", "version"], "Docker Compose"),
        (["node", "--version"],    "Node.js"),
        (["npm", "--version"],     "npm"),
    ]:
        out, err, code = _run_cmd(cmd, cwd=cwd, timeout=5)
        value = out.strip() or err.strip()
        status = "✅" if code == 0 else "❌"
        sections.append(f"  {status} {label}: {value[:60] if value else 'não instalado'}")

    # Arquivos de config presentes
    sections.append("\n📁 Arquivos de configuração:")
    config_files = [
        "requirements.txt", "requirements-dev.txt", "pyproject.toml",
        "setup.py", "setup.cfg", "Dockerfile", "docker-compose.yml",
        "docker-compose.yaml", ".env", ".env.example",
        "Makefile", ".github/workflows",
    ]
    for f in config_files:
        path = Path(cwd) / f
        exists = path.exists()
        sections.append(f"  {'✅' if exists else '  '} {f}")

    # Variáveis de ambiente relevantes (sem revelar valores)
    sections.append("\n🔐 Variáveis de ambiente (presença):")
    env_vars = ["DATABASE_URL", "SECRET_KEY", "API_KEY", "DEBUG",
                "ENVIRONMENT", "PORT", "HOST", "REDIS_URL"]
    for var in env_vars:
        present = var in os.environ
        sections.append(f"  {'✅' if present else '  '} {var}")

    # Pacotes Python instalados (resumo)
    out, _, code = _run_cmd(
        ["python", "-m", "pip", "list", "--format=columns"],
        cwd=cwd, timeout=10,
    )
    if code == 0:
        lines = out.strip().splitlines()
        sections.append(f"\n📦 Pacotes instalados: {max(0, len(lines)-2)}")

    return "\n".join(sections)


@tool
def run_pip(
    action: str,
    packages: Optional[list[str]] = None,
    requirements_file: Optional[str] = None,
    dev: bool = False,
    repo_path: Optional[str] = None,
) -> str:
    """
    Gerencia pacotes Python com pip.

    Args:
        action:            "install", "uninstall", "list", "freeze", "check".
        packages:          Lista de pacotes (ex: ["fastapi==0.110.0", "pydantic"]).
        requirements_file: Arquivo de requirements para instalar (ex: "requirements.txt").
        dev:               Se True, instala dependências de desenvolvimento.
        repo_path:         Diretório raiz do repositório.

    Returns:
        Output do pip com status da operação.
    """
    cwd = repo_path or str(ALLOWED_BASE_PATH)

    if action == "install":
        args = ["python", "-m", "pip", "install"]
        if requirements_file:
            args += ["-r", requirements_file]
        elif packages:
            args += packages
        else:
            return "[ERRO] Forneça packages ou requirements_file para install."

    elif action == "uninstall":
        if not packages:
            return "[ERRO] Forneça packages para uninstall."
        args = ["python", "-m", "pip", "uninstall", "-y"] + packages

    elif action == "list":
        args = ["python", "-m", "pip", "list", "--format=columns"]

    elif action == "freeze":
        args = ["python", "-m", "pip", "freeze"]

    elif action == "check":
        args = ["python", "-m", "pip", "check"]

    else:
        return f"[ERRO] Ação desconhecida: '{action}'. Use: install, uninstall, list, freeze, check."

    stdout, stderr, code = _run_cmd(args, cwd=cwd)
    output = (stdout + stderr).strip()

    status = "✅" if code == 0 else "❌"
    header = f"{status} pip {action}"

    if action == "freeze" and code == 0:
        # Sugere salvar em requirements.txt
        lines = output.splitlines()
        return f"{header} ({len(lines)} pacotes)\n{output}"

    return f"{header}\n{output}" if output else f"{header} — sem output"


@tool
def check_dependencies(
    repo_path: Optional[str] = None,
) -> str:
    """
    Analisa requirements.txt ou pyproject.toml e detecta:
    - Dependências sem versão pinada
    - Conflitos de versão
    - Pacotes desatualizados (se pip list --outdated disponível)
    - Dependências de dev misturadas com prod

    Args:
        repo_path: Diretório raiz do repositório.

    Returns:
        Relatório de saúde das dependências.
    """
    cwd  = repo_path or str(ALLOWED_BASE_PATH)
    root = Path(cwd)

    issues:   list[str] = []
    warnings: list[str] = []
    infos:    list[str] = []

    # Analisa requirements.txt
    req_file = root / "requirements.txt"
    if req_file.exists():
        lines = req_file.read_text().splitlines()
        pkgs  = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
        unpinned = [p for p in pkgs if "==" not in p and ">=" not in p and "~=" not in p]

        infos.append(f"📄 requirements.txt: {len(pkgs)} pacotes")
        if unpinned:
            warnings.append(
                f"⚠️  {len(unpinned)} pacote(s) sem versão pinada:\n"
                + "\n".join(f"   - {p}" for p in unpinned[:10])
            )

    # Analisa pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        infos.append("📄 pyproject.toml encontrado")
        if "[tool.ruff]" not in content:
            warnings.append("⚠️  ruff não configurado em pyproject.toml")
        if "[tool.mypy]" not in content:
            warnings.append("⚠️  mypy não configurado em pyproject.toml")
        if "[tool.pytest" not in content:
            warnings.append("⚠️  pytest não configurado em pyproject.toml")

    # Verifica pacotes desatualizados
    out, _, code = _run_cmd(
        ["python", "-m", "pip", "list", "--outdated", "--format=columns"],
        cwd=cwd, timeout=30,
    )
    if code == 0 and out.strip():
        lines = out.strip().splitlines()[2:]  # remove header
        if lines:
            warnings.append(
                f"📦 {len(lines)} pacote(s) desatualizado(s):\n"
                + "\n".join(f"   {l}" for l in lines[:8])
                + (f"\n   ... e mais {len(lines)-8}" if len(lines) > 8 else "")
            )

    # Verifica conflitos
    out, _, code = _run_cmd(["python", "-m", "pip", "check"], cwd=cwd, timeout=15)
    if code != 0 and out.strip():
        issues.append(f"❌ Conflitos de dependência:\n{out.strip()[:400]}")

    if not req_file.exists() and not pyproject.exists():
        issues.append("❌ Nenhum arquivo de dependências encontrado (requirements.txt ou pyproject.toml)")

    all_lines = infos + warnings + issues
    if not all_lines:
        return "✅ Dependências em ordem. Nenhum problema detectado."

    return "\n".join(all_lines)


@tool
def run_docker(
    action: str,
    args: Optional[list[str]] = None,
    repo_path: Optional[str] = None,
) -> str:
    """
    Executa comandos Docker no repositório.

    Args:
        action:    "build", "run", "compose-up", "compose-down",
                   "compose-build", "ps", "images", "logs".
        args:      Argumentos adicionais para o comando.
        repo_path: Diretório raiz do repositório.

    Returns:
        Output do comando Docker.
    """
    cwd        = repo_path or str(ALLOWED_BASE_PATH)
    extra_args = args or []

    cmd_map = {
        "build":         ["docker", "build", "."] + extra_args,
        "run":           ["docker", "run"] + extra_args,
        "compose-up":    ["docker", "compose", "up", "-d"] + extra_args,
        "compose-down":  ["docker", "compose", "down"] + extra_args,
        "compose-build": ["docker", "compose", "build"] + extra_args,
        "ps":            ["docker", "ps"],
        "images":        ["docker", "images"],
        "logs":          ["docker", "compose", "logs", "--tail=50"] + extra_args,
    }

    if action not in cmd_map:
        return (
            f"[ERRO] Ação desconhecida: '{action}'.\n"
            f"Disponíveis: {', '.join(cmd_map.keys())}"
        )

    # Verifica se Docker está disponível
    _, _, check_code = _run_cmd(["docker", "info"], cwd=cwd, timeout=5)
    if check_code != 0:
        return "[AVISO] Docker não está rodando ou não está instalado."

    stdout, stderr, code = _run_cmd(cmd_map[action], cwd=cwd, timeout=120)
    output = (stdout + stderr).strip()

    status = "✅" if code == 0 else "❌"
    return f"{status} docker {action}\n{'─'*40}\n{output}" if output else f"{status} docker {action} — sem output"


@tool
def scan_ports(
    ports: Optional[list[int]] = None,
) -> str:
    """
    Verifica quais portas estão em uso no sistema.
    Útil para detectar conflitos antes de subir serviços.

    Args:
        ports: Lista de portas para verificar. Se None, verifica portas comuns
               (3000, 5000, 5432, 6379, 8000, 8080, 8888, 27017).

    Returns:
        Status de cada porta: em uso ou livre.
    """
    import socket

    check_ports = ports or [3000, 5000, 5432, 6379, 8000, 8080, 8888, 27017]
    results: list[str] = ["🔌 Status das portas:"]

    port_services = {
        3000: "React/Node", 5000: "Flask", 5432: "PostgreSQL",
        6379: "Redis", 8000: "Django/Uvicorn", 8080: "HTTP alt",
        8888: "Jupyter", 27017: "MongoDB",
    }

    for port in check_ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                in_use = s.connect_ex(("127.0.0.1", port)) == 0

            service = port_services.get(port, "")
            label   = f"{port} ({service})" if service else str(port)
            status  = "🔴 EM USO" if in_use else "🟢 livre"
            results.append(f"  {status}  {label}")
        except Exception:
            results.append(f"  ❓ erro    {port}")

    return "\n".join(results)


@tool
def run_make(
    target: str = "help",
    repo_path: Optional[str] = None,
) -> str:
    """
    Executa um target de um Makefile no repositório.

    Args:
        target:    Target do Makefile (ex: "install", "test", "lint", "run").
        repo_path: Diretório raiz do repositório.

    Returns:
        Output do make.
    """
    cwd = repo_path or str(ALLOWED_BASE_PATH)

    makefile = Path(cwd) / "Makefile"
    if not makefile.exists():
        return (
            "[AVISO] Nenhum Makefile encontrado.\n"
            "Use write_file para criar um com os targets padrão."
        )

    stdout, stderr, code = _run_cmd(["make", target], cwd=cwd)
    output = (stdout + stderr).strip()

    status = "✅" if code == 0 else "❌"
    return f"{status} make {target}\n{'─'*40}\n{output}" if output else f"{status} make {target}"


# ─────────────────────────────────────────────────────────────────────────────
# Todas as tools do DevOps
# ─────────────────────────────────────────────────────────────────────────────

DEVOPS_SHELL_TOOLS = [
    check_environment,
    run_pip,
    check_dependencies,
    run_docker,
    scan_ports,
    run_make,
]

try:
    from tools.git_tools import DEVOPS_GIT_TOOLS
except ImportError:
    DEVOPS_GIT_TOOLS = []

ALL_DEVOPS_TOOLS = DEVOPS_SHELL_TOOLS + DEVOPS_GIT_TOOLS + DEVOPS_TOOLS

# ─────────────────────────────────────────────────────────────────────────────
# Construção do agente
# ─────────────────────────────────────────────────────────────────────────────

_devops_agent_instance = None

def _get_devops_agent():
    global _devops_agent_instance
    if _devops_agent_instance is None:
        llm = make_llm("devops", temperature=0, max_tokens=4096)
        _devops_agent_instance = create_react_agent(
            model=llm,
            tools=ALL_DEVOPS_TOOLS,
            state_modifier=SystemMessage(content=DEVOPS_SYSTEM_PROMPT),
        )
    return _devops_agent_instance


# ─────────────────────────────────────────────────────────────────────────────
# Nó do grafo
# ─────────────────────────────────────────────────────────────────────────────

def devops_node(state: AgentState) -> AgentState:
    """
    Nó do DevOps Agent no grafo LangGraph.

    Gerencia infraestrutura, dependências e configurações de ambiente
    do repositório conforme instrução do supervisor.
    """
    instruction = state.get("current_instruction", "")
    repo_path   = state.get("repo_path", ".")
    task        = state.get("task", "")

    user_prompt = f"""\
## TASK ORIGINAL
{task}

## SUA INSTRUÇÃO (do IT Manager)
{instruction}

## REPOSITÓRIO
{repo_path}

Comece com check_environment para entender o estado atual do ambiente,
depois execute a instrução. Sempre leia os arquivos de configuração
existentes antes de criar ou modificar qualquer um.
"""

    try:
        agent  = _get_devops_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_prompt}]
        })

        final_message = result["messages"][-1]
        output = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )

        status    = _infer_devops_status(output)
        artifacts = _extract_devops_artifacts(output)

    except Exception as e:
        output    = f"❌ Erro no DevOps Agent: {type(e).__name__}: {e}"
        status    = "error"
        artifacts = {}

    updates = record_agent_output(
        state=state,
        agent_name="devops",
        output=output,
        status=status,
        artifacts=artifacts,
    )
    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_devops_status(output: str) -> str:
    lower = output.lower()
    if any(k in lower for k in ["erro", "error", "failed", "❌", "conflito"]):
        return "error"
    if any(k in lower for k in ["aviso", "warning", "⚠️", "desatualizado", "sem versão"]):
        return "warning"
    return "success"


def _extract_devops_artifacts(output: str) -> dict:
    import re
    artifacts: dict = {}

    # Arquivos de config criados/modificados
    config_files = re.findall(
        r'`([^`]+\.(?:txt|toml|yaml|yml|env|cfg|ini|json|Dockerfile|Makefile))`',
        output,
    )
    if config_files:
        artifacts["config_files_changed"] = list(set(config_files))

    # Pacotes instalados
    packages = re.findall(r'pip install[^\n]+', output, re.IGNORECASE)
    if packages:
        artifacts["packages_installed"] = packages

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Exportações
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["devops_node", "DEVOPS_SHELL_TOOLS", "ALL_DEVOPS_TOOLS"]