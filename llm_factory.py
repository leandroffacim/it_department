"""
llm_factory.py
─────────────────────────────────────────────────────────────────────────────
Fábrica central de LLMs para o IT Department Multi-Agent System.

Suporta dois providers via variável de ambiente ITDEPT_LLM_PROVIDER:
  • "anthropic" (padrão) — usa Claude via langchain-anthropic
  • "ollama"              — usa modelos locais via langchain-ollama

Configuração via variáveis de ambiente:

  Provider:
    ITDEPT_LLM_PROVIDER      = anthropic | ollama   (padrão: anthropic)

  Modelos por papel (funciona para ambos os providers):
    ITDEPT_SUPERVISOR_MODEL  = nome do modelo para o Supervisor
    ITDEPT_DEVELOPER_MODEL   = nome do modelo para o Developer
    ITDEPT_QA_MODEL          = nome do modelo para o QA
    ITDEPT_REVIEWER_MODEL    = nome do modelo para o Reviewer
    ITDEPT_DEVOPS_MODEL      = nome do modelo para o DevOps
    ITDEPT_DOCS_MODEL        = nome do modelo para o Docs

  Ollama específico:
    ITDEPT_OLLAMA_BASE_URL   = http://localhost:11434  (padrão)

Exemplos de uso:

  # Anthropic (padrão)
  export ANTHROPIC_API_KEY="sk-ant-..."
  python main.py "minha task" --repo ./repo

  # Ollama com llama3
  export ITDEPT_LLM_PROVIDER=ollama
  export ITDEPT_SUPERVISOR_MODEL=llama3.1:8b
  export ITDEPT_DEVELOPER_MODEL=llama3.1:8b
  export ITDEPT_QA_MODEL=llama3.1:8b
  export ITDEPT_REVIEWER_MODEL=llama3.1:8b
  export ITDEPT_DEVOPS_MODEL=llama3.1:8b
  export ITDEPT_DOCS_MODEL=llama3.1:8b
  python main.py "minha task" --repo ./repo

  # Ollama com modelos diferentes por papel
  export ITDEPT_LLM_PROVIDER=ollama
  export ITDEPT_SUPERVISOR_MODEL=llama3.1:70b   # mais capaz para planejar
  export ITDEPT_DEVELOPER_MODEL=qwen2.5-coder:7b  # especializado em código
  export ITDEPT_QA_MODEL=qwen2.5-coder:7b
  export ITDEPT_REVIEWER_MODEL=llama3.1:8b
  export ITDEPT_DEVOPS_MODEL=llama3.1:8b
  export ITDEPT_DOCS_MODEL=llama3.1:8b

Modelos Ollama recomendados para este projeto:
  Geral:   llama3.1:8b, llama3.1:70b, mistral:7b, qwen2.5:7b
  Código:  qwen2.5-coder:7b, qwen2.5-coder:14b, deepseek-coder-v2:16b
  Leve:    llama3.2:3b, phi3:mini (para máquinas com pouca RAM)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("it_department.llm_factory")

# ─────────────────────────────────────────────────────────────────────────────
# Defaults por papel e provider
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER = os.environ.get("ITDEPT_LLM_PROVIDER", "anthropic").lower().strip()

DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "supervisor": "claude-opus-4-5",
        "developer":  "claude-sonnet-4-5",
        "qa":         "claude-sonnet-4-5",
        "reviewer":   "claude-sonnet-4-5",
        "devops":     "claude-sonnet-4-5",
        "docs":       "claude-sonnet-4-5",
    },
    "ollama": {
        "supervisor": "llama3.1:8b",
        "developer":  "qwen2.5-coder:7b",
        "qa":         "qwen2.5-coder:7b",
        "reviewer":   "llama3.1:8b",
        "devops":     "llama3.1:8b",
        "docs":       "llama3.1:8b",
    },
}

OLLAMA_BASE_URL = os.environ.get("ITDEPT_OLLAMA_BASE_URL", "http://localhost:11434")


# ─────────────────────────────────────────────────────────────────────────────
# Resolução de modelo
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_model(role: str) -> str:
    """
    Resolve o nome do modelo para um papel específico.
    Ordem: env var específica → default do provider.
    """
    env_key = f"ITDEPT_{role.upper()}_MODEL"
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val

    provider_defaults = DEFAULTS.get(PROVIDER, DEFAULTS["ollama"])
    return provider_defaults.get(role, provider_defaults.get("developer", "llama3.1:8b"))


# ─────────────────────────────────────────────────────────────────────────────
# Factory principal
# ─────────────────────────────────────────────────────────────────────────────

def make_llm(role: str, temperature: float = 0, max_tokens: int = 4096) -> Any:
    """
    Cria e retorna um LLM configurado para o papel especificado.

    Args:
        role:        Papel do agente: "supervisor", "developer", "qa",
                     "reviewer", "devops", "docs".
        temperature: Temperatura do modelo (0 = determinístico).
        max_tokens:  Máximo de tokens na resposta.

    Returns:
        Instância do LLM compatível com LangChain (ChatAnthropic ou ChatOllama).

    Raises:
        ImportError: Se o pacote do provider não estiver instalado.
        ValueError:  Se o provider configurado for desconhecido.
    """
    model = _resolve_model(role)

    if PROVIDER == "anthropic":
        return _make_anthropic(model, temperature, max_tokens)
    elif PROVIDER == "ollama":
        return _make_ollama(model, temperature, max_tokens)
    else:
        raise ValueError(
            f"Provider desconhecido: '{PROVIDER}'. "
            f"Use ITDEPT_LLM_PROVIDER=anthropic ou ITDEPT_LLM_PROVIDER=ollama"
        )


def _make_anthropic(model: str, temperature: float, max_tokens: int) -> Any:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic não instalado.\n"
            "Execute: pip install langchain-anthropic"
        )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY não definida.\n"
            "Execute: export ANTHROPIC_API_KEY='sua-chave'"
        )

    logger.debug("LLM Anthropic: role=%s model=%s", "?", model)
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _make_ollama(model: str, temperature: float, max_tokens: int) -> Any:
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama não instalado.\n"
            "Execute: pip install langchain-ollama"
        )

    logger.debug("LLM Ollama: model=%s base_url=%s", model, OLLAMA_BASE_URL)
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=max_tokens,   # Ollama usa num_predict em vez de max_tokens
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnóstico
# ─────────────────────────────────────────────────────────────────────────────

def print_config() -> None:
    """Imprime a configuração atual do LLM factory."""
    roles = ["supervisor", "developer", "qa", "reviewer", "devops", "docs"]

    print(f"\n  Provider:  {PROVIDER}")
    if PROVIDER == "ollama":
        print(f"  Base URL:  {OLLAMA_BASE_URL}")
    print()

    for role in roles:
        model   = _resolve_model(role)
        env_key = f"ITDEPT_{role.upper()}_MODEL"
        source  = "env" if os.environ.get(env_key) else "default"
        print(f"  {role:12s}  {model:35s}  [{source}]")


def check_ollama_connection() -> bool:
    """
    Verifica se o servidor Ollama está acessível.
    Retorna True se conectou, False caso contrário.
    """
    import urllib.request
    try:
        req = urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return req.status == 200
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """Lista os modelos disponíveis no servidor Ollama local."""
    import json
    import urllib.request
    try:
        req  = urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = json.loads(req.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# CLI de diagnóstico
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔧 IT Department — LLM Factory Config")
    print("─" * 45)
    print_config()

    if PROVIDER == "ollama":
        print("\n🔌 Conexão Ollama:")
        ok = check_ollama_connection()
        print(f"  {'✅ Conectado' if ok else '❌ Sem conexão'}  ({OLLAMA_BASE_URL})")

        if ok:
            models = list_ollama_models()
            print(f"\n  Modelos disponíveis ({len(models)}):")
            for m in models:
                print(f"    • {m}")

            # Avisa se algum modelo configurado não está disponível
            roles = ["supervisor", "developer", "qa", "reviewer", "devops", "docs"]
            missing = []
            for role in roles:
                model = _resolve_model(role)
                if model not in models:
                    missing.append((role, model))

            if missing:
                print(f"\n  ⚠️  Modelos não encontrados localmente:")
                for role, model in missing:
                    print(f"    {role:12s} → {model}")
                    print(f"              Baixe com: ollama pull {model}")
        else:
            print(f"\n  Inicie o Ollama com: ollama serve")
    print()