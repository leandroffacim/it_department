"""
state.py
─────────────────────────────────────────────────────────────────────────────
Estado compartilhado entre todos os agentes do IT Department.
Cada nó lê e escreve neste TypedDict — é a memória de trabalho do grafo.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    # ── Input do usuário ───────────────────────────────────────────────────
    task: str
    """Request original do usuário (ex: 'adicionar autenticação JWT')."""

    repo_path: str
    """Caminho absoluto ou relativo do repositório local a trabalhar."""

    # ── Histórico de mensagens (acumulativo) ───────────────────────────────
    messages: Annotated[list[BaseMessage], operator.add]
    """Histórico completo da conversa entre user, supervisor e agentes.
    Usa operator.add para que múltiplos nós possam fazer append sem conflito."""

    # ── Controle de fluxo (supervisor) ────────────────────────────────────
    plan: str
    """Plano de execução criado pelo supervisor na primeira iteração."""

    next_agent: str
    """Próximo agente a executar: 'developer' | 'qa' | 'reviewer' | 'devops' | 'docs' | 'FINISH'."""

    current_instruction: str
    """Instrução detalhada do supervisor para o agente atual."""

    iteration: int
    """Contador de iterações — protege contra loops infinitos."""

    routing_history: list[dict]
    """Log de cada decisão de roteamento: [{agent, reason, timestamp, iteration}]."""

    # ── Outputs dos agentes ───────────────────────────────────────────────
    agent_outputs: Annotated[list[dict], operator.add]
    """Lista de resultados de cada agente: [{agent, output, status, timestamp}].
    Usa operator.add para acumulação segura em paralelo."""

    artifacts: dict[str, Any]
    """Artefatos produzidos durante a execução.
    Ex: {"files_changed": [...], "test_report": "...", "coverage": 87.3}"""

    # ── Human-in-the-loop ─────────────────────────────────────────────────
    human_approved: Optional[bool]
    """Se True, o humano aprovou o estado atual para continuar.
    None = ainda não revisado. False = rejeitado com feedback."""

    human_feedback: Optional[str]
    """Feedback textual do humano ao revisar (usado quando human_approved=False)."""

    # ── Resultado final ───────────────────────────────────────────────────
    final_summary: Optional[str]
    """Resumo do que foi feito, gerado pelo supervisor ao finalizar."""
