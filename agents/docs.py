"""
docs.py
─────────────────────────────────────────────────────────────────────────────
Agente Docs Writer do IT Department Multi-Agent System.

Responsabilidades:
  • Gerar e atualizar README.md
  • Inserir/atualizar docstrings (Google style)
  • Gerar CHANGELOG a partir do git log
  • Criar diagramas de arquitetura em Mermaid
  • Documentar APIs (endpoints, parâmetros, exemplos)
  • Manter .env.example sincronizado

Ferramentas:
  doc_tools  → generate_readme, generate_docstrings, generate_changelog,
               generate_mermaid, generate_env_example
  fs_tools   → read_file, write_file, append_file, get_repo_tree, search_in_files
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from llm_factory import make_llm
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from state import AgentState
from supervisor import record_agent_output
from tools.filesystem import DOCS_TOOLS

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

DOCS_MODEL = os.environ.get("ITDEPT_DOCS_MODEL", "claude-sonnet-4-5")

try:
    from tools.filesystem import ALLOWED_BASE_PATH
except ImportError:
    ALLOWED_BASE_PATH = Path(os.environ.get("ITDEPT_BASE_PATH", str(Path.cwd()))).resolve()

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

DOCS_SYSTEM_PROMPT = """\
Você é um Technical Writer sênior especializado em documentação de software Python.
Sua missão é produzir documentação clara, precisa e que desenvolvedores realmente usem.

## SUAS FERRAMENTAS

- generate_readme        → gera estrutura completa de README.md
- generate_docstrings    → extrai funções sem docstring de um arquivo
- generate_changelog     → gera CHANGELOG a partir do git log
- generate_mermaid       → cria diagrama de arquitetura do projeto
- generate_env_example   → analisa o código e gera .env.example
- read_file              → leia arquivos para entender o contexto
- write_file             → crie ou substitua documentação
- append_file            → adicione conteúdo a docs existentes
- get_repo_tree          → mapa do repositório para documentar estrutura
- search_in_files        → busque padrões (ex: endpoint routes, env vars)

## PRINCÍPIOS DE BOA DOCUMENTAÇÃO

1. **README deve responder em 60 segundos:**
   - O que é? Uma frase.
   - Como instalar? 3 comandos.
   - Como usar? Um exemplo mínimo funcional.
   - Onde tem mais info? Links.

2. **Docstrings Google Style:**
   ```python
   def process(data: list[str], limit: int = 10) -> dict:
       \"\"\"Processa uma lista de strings e retorna métricas.

       Args:
           data:  Lista de strings para processar.
           limit: Número máximo de itens a considerar.

       Returns:
           Dicionário com métricas: count, unique, truncated.

       Raises:
           ValueError: Se data estiver vazia.

       Example:
           >>> process(["a", "b", "a"])
           {'count': 3, 'unique': 2, 'truncated': False}
       \"\"\"
   ```

3. **Changelog semântico (Keep a Changelog):**
   - Seções: Added, Changed, Deprecated, Removed, Fixed, Security
   - Formato: ## [versão] - YYYY-MM-DD

4. **Diagramas Mermaid:**
   - Use flowchart LR para arquitetura
   - Use sequenceDiagram para fluxos de request
   - Use classDiagram para modelos de dados

## O QUE NUNCA FAZER

- Documentação óbvia: `# incrementa i` para `i += 1`
- Copy-paste do código na doc
- Docs desatualizadas (sempre sincronize com o código real)
- README com 10 badges e zero exemplos funcionais

## REPORT FINAL

```
## Documentação gerada/atualizada
- <arquivo>: <descrição do que foi criado>

## Cobertura de docstrings
- Antes: X funções sem doc | Depois: Y funções sem doc

## Próximos passos sugeridos
- <o que ainda falta documentar>
```
"""

# ─────────────────────────────────────────────────────────────────────────────
# Doc Generation Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def generate_readme(
    repo_path: Optional[str] = None,
) -> str:
    """
    Analisa o repositório e gera um esqueleto estruturado de README.md.
    O agente usa isso como base e preenche com informações reais do projeto.

    Coleta automaticamente: nome do projeto, estrutura de pastas,
    arquivos de config presentes, entrypoints, e dependências principais.

    Args:
        repo_path: Diretório raiz do repositório.

    Returns:
        Template de README.md preenchido com informações reais do projeto.
    """
    root = Path(repo_path or ALLOWED_BASE_PATH)

    # Coleta informações do projeto
    project_name = root.name
    has_docker   = (root / "Dockerfile").exists()
    has_make     = (root / "Makefile").exists()
    has_tests    = any(root.rglob("test_*.py")) or (root / "tests").exists()

    # Dependências principais
    deps: list[str] = []
    req_file = root / "requirements.txt"
    if req_file.exists():
        lines = req_file.read_text().splitlines()
        deps  = [l.split("==")[0].strip() for l in lines
                 if l.strip() and not l.startswith("#")][:8]

    pyproject = root / "pyproject.toml"
    if pyproject.exists() and not deps:
        content = pyproject.read_text()
        found   = re.findall(r'"([a-zA-Z][a-zA-Z0-9_-]+)(?:[>=<!]|")', content)
        deps    = found[:8]

    # Entrypoint principal
    entrypoints = []
    for candidate in ["main.py", "app.py", "run.py", "server.py", "cli.py"]:
        if (root / candidate).exists():
            entrypoints.append(candidate)

    # Monta template
    deps_section = ""
    if deps:
        deps_section = "## Dependências Principais\n\n" + " ".join(
            f"`{d}`" for d in deps
        ) + "\n\n"

    install_cmd = "pip install -r requirements.txt"
    if (root / "pyproject.toml").exists():
        install_cmd = "pip install -e ."

    run_cmd = f"python {entrypoints[0]}" if entrypoints else "python -m <modulo>"
    if has_make:
        run_cmd = "make run"

    docker_section = ""
    if has_docker:
        docker_section = """## Docker

```bash
docker build -t {name} .
docker run -p 8000:8000 {name}
```

""".format(name=project_name.lower().replace(" ", "-"))

    test_section = ""
    if has_tests:
        test_section = """## Testes

```bash
{cmd}
```

""".format(cmd="make test" if has_make else "pytest")

    template = f"""# {project_name}

> Breve descrição do projeto em uma frase.

## Instalação

```bash
git clone <repo-url>
cd {project_name}
{install_cmd}
cp .env.example .env   # configure suas variáveis
```

## Uso

```bash
{run_cmd}
```

**Exemplo mínimo:**

```python
# TODO: adicionar exemplo de uso real
```

{deps_section}{docker_section}{test_section}## Estrutura do Projeto

```
{project_name}/
├── ...   # use get_repo_tree para preencher
```

## Contribuindo

1. Fork o projeto
2. Crie uma branch: `git checkout -b feat/minha-feature`
3. Commit: `git commit -m 'feat: descrição'`
4. Push: `git push origin feat/minha-feature`
5. Abra um Pull Request

## Licença

MIT
"""
    return template


@tool
def generate_docstrings(
    path: str,
    repo_path: Optional[str] = None,
) -> str:
    """
    Analisa um arquivo Python e lista todas as funções/métodos sem docstring,
    com sua assinatura completa. O agente usa essa info para escrever as docs.

    Args:
        path:      Arquivo Python para analisar.
        repo_path: Diretório raiz do repositório.

    Returns:
        Lista de funções sem docstring com assinatura, tipo de retorno e
        template de docstring Google style para cada uma.
    """
    cwd      = Path(repo_path or ALLOWED_BASE_PATH)
    filepath = (cwd / path).resolve()

    if not filepath.exists():
        return f"[ERRO] Arquivo não encontrado: '{path}'"
    if filepath.suffix != ".py":
        return "[ERRO] generate_docstrings funciona apenas com arquivos .py"

    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree   = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return f"[ERRO] Sintaxe inválida: {e}"

    missing: list[dict] = []
    has_doc: int = 0

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Verifica se tem docstring
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            has_doc += 1
            continue

        # Coleta assinatura
        args     = [a.arg for a in node.args.args]
        defaults = [ast.unparse(d) for d in node.args.defaults]
        # Alinha defaults com os últimos args
        n_without_default = len(args) - len(defaults)
        arg_parts = []
        for i, arg in enumerate(args):
            default_idx = i - n_without_default
            if default_idx >= 0:
                arg_parts.append(f"{arg}={defaults[default_idx]}")
            else:
                arg_parts.append(arg)

        # Anotações
        annotations = {}
        for a in node.args.args:
            if a.annotation:
                try:
                    annotations[a.arg] = ast.unparse(a.annotation)
                except Exception:
                    pass

        # Retorno
        return_ann = ""
        if node.returns:
            try:
                return_ann = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        sig = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}({', '.join(arg_parts)}){return_ann}"

        # Template de docstring
        args_section = ""
        real_args = [a for a in args if a not in ("self", "cls")]
        if real_args:
            args_lines = []
            for a in real_args:
                type_hint = annotations.get(a, "")
                type_str  = f" ({type_hint})" if type_hint else ""
                args_lines.append(f"            {a}{type_str}: Descrição de {a}.")
            args_section = "        Args:\n" + "\n".join(args_lines) + "\n\n"

        returns_section = ""
        if return_ann and "None" not in return_ann:
            returns_section = f"        Returns:\n            Descrição do retorno{return_ann}.\n\n"

        docstring_template = (
            f'        """Descrição concisa do que {node.name} faz.\n\n'
            f"{args_section}"
            f"{returns_section}"
            f'        """'
        )

        missing.append({
            "name":     node.name,
            "line":     node.lineno,
            "sig":      sig,
            "template": docstring_template,
        })

    if not missing:
        return (
            f"✅ Todas as {has_doc} funções em '{path}' já têm docstring!\n"
            f"   Nenhuma ação necessária."
        )

    total = has_doc + len(missing)
    pct   = int(100 * has_doc / total) if total else 0
    lines = [
        f"📝 Docstrings em '{path}': {has_doc}/{total} ({pct}% cobertura)",
        f"   {len(missing)} função(ões) precisam de docstring:",
        "─" * 55,
    ]

    for item in missing:
        lines.append(f"\n  📌 linha {item['line']}: {item['sig']}")
        lines.append(f"     Template sugerido:")
        lines.append(item["template"])

    return "\n".join(lines)


@tool
def generate_changelog(
    version: str = "Unreleased",
    since_tag: Optional[str] = None,
    max_commits: int = 50,
    repo_path: Optional[str] = None,
) -> str:
    """
    Gera uma entrada de CHANGELOG no formato Keep a Changelog
    a partir do histórico de commits git.

    Args:
        version:     Versão para esta entrada (ex: "1.2.0", "Unreleased").
        since_tag:   Tag git de início (ex: "v1.1.0"). Se None, usa últimos commits.
        max_commits: Número máximo de commits a incluir.
        repo_path:   Diretório raiz do repositório.

    Returns:
        Entrada de CHANGELOG formatada, pronta para inserir no arquivo.
    """
    cwd = repo_path or str(ALLOWED_BASE_PATH)

    # Coleta commits
    git_args = ["git", "log", f"-{max_commits}",
                "--pretty=format:%s|%h|%an"]
    if since_tag:
        git_args = ["git", "log", f"{since_tag}..HEAD",
                    "--pretty=format:%s|%h|%an"]

    try:
        result = subprocess.run(
            git_args, cwd=cwd, capture_output=True, text=True, timeout=15,
        )
        raw_commits = result.stdout.strip().splitlines()
    except Exception:
        raw_commits = []

    if not raw_commits:
        return (
            "[AVISO] Nenhum commit encontrado.\n"
            "Verifique se o repositório tem commits e se since_tag existe."
        )

    # Classifica commits por tipo semântico
    categories: dict[str, list[str]] = {
        "Added":      [],
        "Changed":    [],
        "Fixed":      [],
        "Removed":    [],
        "Security":   [],
        "Deprecated": [],
        "Other":      [],
    }

    type_map = {
        "feat":     "Added",
        "fix":      "Fixed",
        "refactor": "Changed",
        "perf":     "Changed",
        "style":    "Changed",
        "docs":     "Changed",
        "chore":    "Other",
        "test":     "Other",
        "ci":       "Other",
        "build":    "Other",
        "security": "Security",
        "deprecat": "Deprecated",
        "remove":   "Removed",
        "revert":   "Changed",
    }

    for line in raw_commits:
        parts   = line.split("|")
        subject = parts[0].strip() if parts else line
        short   = parts[1].strip() if len(parts) > 1 else ""

        # Detecta tipo semântico
        category = "Other"
        match = re.match(r'^(\w+)(?:\(.+?\))?[!:]?\s*', subject)
        if match:
            commit_type = match.group(1).lower()
            category = type_map.get(commit_type, "Other")

        # Limpa a mensagem
        clean = re.sub(r'^(\w+)(?:\(.+?\))?[!:]?\s*', '', subject).strip()
        if not clean:
            clean = subject

        entry = f"- {clean.capitalize()} ({short})" if short else f"- {clean.capitalize()}"
        categories[category].append(entry)

    # Monta o bloco de changelog
    today   = datetime.now().strftime("%Y-%m-%d")
    header  = f"## [{version}] - {today}"
    blocks  = [header]
    order   = ["Added", "Changed", "Fixed", "Security", "Deprecated", "Removed", "Other"]

    for cat in order:
        entries = categories[cat]
        if entries:
            blocks.append(f"\n### {cat}")
            blocks.extend(entries)

    result_text = "\n".join(blocks)

    # Verifica se já existe CHANGELOG.md
    changelog_path = Path(cwd) / "CHANGELOG.md"
    existing_note  = ""
    if changelog_path.exists():
        existing_note = (
            "\n\n💡 CHANGELOG.md já existe. "
            "Use append_file ou patch_file para inserir esta entrada no topo "
            "(após a linha '# Changelog')."
        )

    return result_text + existing_note


@tool
def generate_mermaid(
    diagram_type: str = "flowchart",
    repo_path: Optional[str] = None,
) -> str:
    """
    Analisa o repositório e gera um diagrama Mermaid da arquitetura.

    Args:
        diagram_type: "flowchart" (estrutura de módulos) ou
                      "classDiagram" (classes e relações).
        repo_path:    Diretório raiz do repositório.

    Returns:
        Código Mermaid do diagrama, pronto para inserir em Markdown.
    """
    root  = Path(repo_path or ALLOWED_BASE_PATH)
    lines = ["```mermaid"]

    if diagram_type == "flowchart":
        lines.append("flowchart LR")

        # Descobre pacotes/módulos principais
        py_dirs: list[Path] = []
        py_files: list[Path] = []

        for item in sorted(root.iterdir()):
            if item.is_dir() and not item.name.startswith(".") \
                    and item.name not in {"__pycache__", "node_modules", ".venv",
                                          "venv", "dist", "build", ".git"}:
                py_dirs.append(item)
            elif item.suffix == ".py" and item.name != "__init__.py":
                py_files.append(item)

        # Nó raiz
        proj = root.name.replace("-", "_").replace(" ", "_")
        lines.append(f"    {proj}[/{root.name}/]")

        # Sub-módulos
        for d in py_dirs[:8]:
            dname = d.name.replace("-", "_")
            lines.append(f"    {dname}[{d.name}/]")
            lines.append(f"    {proj} --> {dname}")

            # Arquivos dentro do dir
            for f in sorted(d.glob("*.py"))[:5]:
                if f.name == "__init__.py":
                    continue
                fname = f"{dname}_{f.stem}".replace("-", "_")
                lines.append(f"    {fname}({f.name})")
                lines.append(f"    {dname} --> {fname}")

        # Arquivos na raiz
        for f in py_files[:5]:
            fname = f.stem.replace("-", "_")
            lines.append(f"    {fname}({f.name})")
            lines.append(f"    {proj} --> {fname}")

    elif diagram_type == "classDiagram":
        lines.append("classDiagram")

        # Coleta classes de todos os arquivos .py
        classes_found: list[tuple[str, list[str]]] = []
        for py_file in sorted(root.rglob("*.py"))[:20]:
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree   = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        methods = [
                            n.name for n in ast.walk(node)
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and not n.name.startswith("__")
                        ][:5]
                        classes_found.append((node.name, methods))
            except Exception:
                continue

        if not classes_found:
            lines = ["```mermaid", "classDiagram", '    note "Nenhuma classe encontrada"']
        else:
            for cls_name, methods in classes_found[:10]:
                lines.append(f"    class {cls_name}{{")
                for m in methods:
                    lines.append(f"        +{m}()")
                lines.append("    }")

    lines.append("```")

    return "\n".join(lines)


@tool
def generate_env_example(
    repo_path: Optional[str] = None,
) -> str:
    """
    Escaneia o código em busca de variáveis de ambiente usadas via
    os.environ, os.getenv, e config patterns, e gera um .env.example
    documentado com descrições e valores de exemplo.

    Args:
        repo_path: Diretório raiz do repositório.

    Returns:
        Conteúdo do .env.example gerado, com comentários descritivos.
    """
    root = Path(repo_path or ALLOWED_BASE_PATH)

    env_vars: dict[str, str] = {}

    # Padrões para detectar uso de env vars
    patterns = [
        r'os\.environ\.get\(["\']([A-Z_][A-Z0-9_]+)["\']',
        r'os\.getenv\(["\']([A-Z_][A-Z0-9_]+)["\']',
        r'os\.environ\[["\']([A-Z_][A-Z0-9_]+)["\']',
        r'environ\.get\(["\']([A-Z_][A-Z0-9_]+)["\']',
    ]

    # Valores de exemplo por categoria
    examples: dict[str, str] = {
        "DATABASE_URL":   "postgresql://user:password@localhost:5432/dbname",
        "REDIS_URL":      "redis://localhost:6379/0",
        "SECRET_KEY":     "your-secret-key-change-in-production",
        "API_KEY":        "your-api-key-here",
        "DEBUG":          "false",
        "PORT":           "8000",
        "HOST":           "0.0.0.0",
        "LOG_LEVEL":      "INFO",
        "ENVIRONMENT":    "development",
        "ALLOWED_HOSTS":  "localhost,127.0.0.1",
        "CORS_ORIGINS":   "http://localhost:3000",
        "JWT_SECRET":     "your-jwt-secret-here",
        "SENTRY_DSN":     "https://your-sentry-dsn",
        "AWS_ACCESS_KEY_ID":     "your-aws-access-key",
        "AWS_SECRET_ACCESS_KEY": "your-aws-secret-key",
        "AWS_REGION":            "us-east-1",
        "SMTP_HOST":      "smtp.gmail.com",
        "SMTP_PORT":      "587",
        "SMTP_USER":      "your@email.com",
        "SMTP_PASSWORD":  "your-email-password",
    }

    descriptions: dict[str, str] = {
        "DATABASE_URL":   "URL de conexão com o banco de dados",
        "REDIS_URL":      "URL de conexão com o Redis",
        "SECRET_KEY":     "Chave secreta da aplicação — gere com: python -c \"import secrets; print(secrets.token_hex(32))\"",
        "API_KEY":        "Chave de API externa",
        "DEBUG":          "Modo debug — NUNCA true em produção",
        "PORT":           "Porta em que a aplicação vai rodar",
        "HOST":           "Host de bind do servidor",
        "LOG_LEVEL":      "Nível de log: DEBUG, INFO, WARNING, ERROR",
        "ENVIRONMENT":    "Ambiente: development, staging, production",
        "JWT_SECRET":     "Secret para assinatura de tokens JWT",
        "SENTRY_DSN":     "DSN do Sentry para monitoramento de erros",
    }

    # Escaneia todos os arquivos Python
    for py_file in sorted(root.rglob("*.py")):
        if any(part in str(py_file) for part in ["__pycache__", ".venv", "venv"]):
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                for match in re.finditer(pattern, content):
                    var = match.group(1)
                    env_vars[var] = examples.get(var, "your-value-here")
        except Exception:
            continue

    if not env_vars:
        return (
            "[INFO] Nenhuma variável de ambiente encontrada no código.\n"
            "Certifique-se de usar os.environ.get('VAR_NAME') para que sejam detectadas."
        )

    # Gera o arquivo
    lines = [
        "# .env.example",
        "# Copie este arquivo para .env e preencha com seus valores reais.",
        "# NUNCA commite o .env — apenas o .env.example.",
        "#",
        f"# Gerado automaticamente em {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]

    # Agrupa por categoria
    categories = {
        "# ── Aplicação ──────────────────────────────────────────────────────────":
            ["DEBUG", "ENVIRONMENT", "PORT", "HOST", "SECRET_KEY", "ALLOWED_HOSTS", "LOG_LEVEL"],
        "# ── Banco de Dados ─────────────────────────────────────────────────────":
            ["DATABASE_URL", "REDIS_URL"],
        "# ── Autenticação ───────────────────────────────────────────────────────":
            ["JWT_SECRET", "API_KEY"],
        "# ── AWS ────────────────────────────────────────────────────────────────":
            ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
        "# ── Email ──────────────────────────────────────────────────────────────":
            ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD"],
        "# ── Monitoramento ──────────────────────────────────────────────────────":
            ["SENTRY_DSN", "CORS_ORIGINS"],
    }

    shown: set[str] = set()

    for category_header, category_vars in categories.items():
        category_found = [v for v in category_vars if v in env_vars]
        if not category_found:
            continue
        lines.append(category_header)
        for var in category_found:
            desc = descriptions.get(var, "")
            if desc:
                lines.append(f"# {desc}")
            lines.append(f"{var}={env_vars[var]}")
            lines.append("")
            shown.add(var)

    # Variáveis não categorizadas
    remaining = [v for v in sorted(env_vars.keys()) if v not in shown]
    if remaining:
        lines.append("# ── Outros ─────────────────────────────────────────────────────────────")
        for var in remaining:
            lines.append(f"{var}={env_vars[var]}")
            lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Todas as tools do Docs Writer
# ─────────────────────────────────────────────────────────────────────────────

DOCS_GEN_TOOLS = [
    generate_readme,
    generate_docstrings,
    generate_changelog,
    generate_mermaid,
    generate_env_example,
]

ALL_DOCS_TOOLS = DOCS_GEN_TOOLS + DOCS_TOOLS  # DOCS_TOOLS = fs tools

# ─────────────────────────────────────────────────────────────────────────────
# Construção do agente
# ─────────────────────────────────────────────────────────────────────────────

_docs_agent_instance = None

def _get_docs_agent():
    global _docs_agent_instance
    if _docs_agent_instance is None:
        llm = make_llm("docs", temperature=0.2, max_tokens=4096)
        _docs_agent_instance = create_react_agent(
            model=llm,
            tools=ALL_DOCS_TOOLS,
            state_modifier=SystemMessage(content=DOCS_SYSTEM_PROMPT),
        )
    return _docs_agent_instance


# ─────────────────────────────────────────────────────────────────────────────
# Nó do grafo
# ─────────────────────────────────────────────────────────────────────────────

def docs_node(state: AgentState) -> AgentState:
    """
    Nó do Docs Writer Agent no grafo LangGraph.

    Gera e atualiza documentação do projeto conforme instrução do supervisor.
    """
    instruction   = state.get("current_instruction", "")
    repo_path     = state.get("repo_path", ".")
    task          = state.get("task", "")
    changed_files = state.get("artifacts", {}).get("files_changed", [])

    changed_ctx = ""
    if changed_files:
        py_files = [f for f in changed_files if f.endswith(".py")]
        if py_files:
            changed_ctx = (
                "\n\n## Arquivos Python modificados (priorize docstrings nestes):\n"
                + "\n".join(f"  - {f}" for f in py_files)
            )

    user_prompt = f"""\
## TASK ORIGINAL
{task}

## SUA INSTRUÇÃO (do IT Manager)
{instruction}
{changed_ctx}

## REPOSITÓRIO
{repo_path}

Gere a documentação necessária. Comece com get_repo_tree para entender
a estrutura do projeto. Use as ferramentas de geração para criar o conteúdo
base e enriqueça com informações reais lidas dos arquivos de código.

Sempre leia os arquivos .py antes de escrever docstrings — você precisa
entender o que o código realmente faz para documentar com precisão.
"""

    try:
        agent  = _get_docs_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_prompt}]
        })

        final_message = result["messages"][-1]
        output = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )

        status    = "success" if "erro" not in output.lower() else "warning"
        artifacts = _extract_docs_artifacts(output)

    except Exception as e:
        output    = f"❌ Erro no Docs Agent: {type(e).__name__}: {e}"
        status    = "error"
        artifacts = {}

    updates = record_agent_output(
        state=state,
        agent_name="docs",
        output=output,
        status=status,
        artifacts=artifacts,
    )
    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_docs_artifacts(output: str) -> dict:
    artifacts: dict = {}

    doc_files = re.findall(
        r'`([^`]+\.(?:md|rst|txt))`', output
    )
    if doc_files:
        artifacts["docs_files_updated"] = list(set(doc_files))

    if "README" in output:
        artifacts["readme_updated"] = True
    if "CHANGELOG" in output:
        artifacts["changelog_updated"] = True
    if "docstring" in output.lower():
        artifacts["docstrings_added"] = True

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Exportações
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["docs_node", "DOCS_GEN_TOOLS", "ALL_DOCS_TOOLS"]