"""Bounded, non-content email context supplied by the Sicurre gateway."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MailContext:
    """Transport/context evidence that may refine spam versus legitimate."""

    structured_forward: bool = False
    outer_sender_authenticated: bool = False
    mailing_list_headers: bool = False
    subscription_claimed: bool = False

    def prompt_summary(self) -> str:
        """Render bounded trusted context without copying message content."""

        return (
            f"transfert_structure={str(self.structured_forward).lower()}; "
            f"expediteur_externe_authentifie="
            f"{str(self.outer_sender_authenticated).lower()}; "
            f"entetes_liste={str(self.mailing_list_headers).lower()}; "
            f"abonnement_revendique={str(self.subscription_claimed).lower()}"
        )
