"""URL-rule regressions for legitimate bulk-email link patterns."""

from src.inference.rules import check_url_rules


def test_separate_long_tracking_links_do_not_accumulate_phishing_risk() -> None:
    links = "\n".join(
        f"https://tracking.example.com/click/{character * 170}"
        for character in ("a", "b", "c")
    )

    result = check_url_rules(links)

    assert result.is_phishing is False
    assert result.risk_score == 20


def test_multiple_signals_on_one_url_still_combine() -> None:
    result = check_url_rules(
        "https://paypal-verify-account.example.xyz/login/"
        + ("a" * 170)
    )

    assert result.is_phishing is True
    assert result.risk_score >= 30
