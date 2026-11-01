from decimal import Decimal

import pytest
from pydantic import ValidationError

from banking import (
    Credit,
    Database,
    PackageQuote,
    QuoteRefusal,
    Savings,
    calculate,
    expected_database,
    make_case,
    quote_package,
    score,
)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 54, 404])
def test_quotes_agree_with_the_independent_scorer_without_selecting_a_package(seed):
    case = make_case(seed)
    wanted = expected_database(case)
    for account, report, credit in zip(case.savings, wanted.reports, wanted.credits, strict=True):
        quotes = []
        for checking in [None] + [a.account_id for a in case.checking]:
            for card in [None] + [a.card_id for a in case.cards]:
                quote = quote_package(case, account.account_id, checking, card)
                if isinstance(quote, PackageQuote):
                    quotes.append(quote)
        assert len(quotes) > 1
        best = max(
            quotes, key=lambda q: (q.annual_bonus, q.card_bonus_percent, q.checking_bonus_percent)
        )
        fields = report.model_dump(exclude={"correction_credit_id"})
        assert {k: getattr(best, k) for k in fields} == fields
        assert best.correction_amount == credit.amount
        base = quote_package(case, account.account_id)
        assert isinstance(base, PackageQuote) and base.annual_bonus == 0


def test_quote_rejects_unknown_ineligible_and_excluded_packages():
    case = make_case(2)
    with pytest.raises(ValueError, match="not in"):
        quote_package(case, "missing")
    refused = [
        quote_package(case, case.savings[0].account_id, case.checking[0].account_id),
        quote_package(case, case.savings[0].account_id, card_id=case.cards[0].card_id),
        quote_package(
            case, case.savings[0].account_id, case.checking[2].account_id, case.cards[2].card_id
        ),
    ]
    assert all(isinstance(q, QuoteRefusal) and not q.permitted for q in refused)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("12000 * 3.4 / 100 / 12", "34.00"),
        ("1.005", "1.01"),
        ("10 - (-2) + 0.5", "12.50"),
        ("(3+5)/2", "4.00"),
    ],
)
def test_calculator(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os')", "2**8", "[1]", "1/0"])
def test_calculator_rejects_other_code(expression):
    with pytest.raises((ValueError, ArithmeticError)):
        calculate(expression)


def test_reproducible_cases_and_strict_scorer():
    case = make_case(42)
    assert case == make_case(42)
    assert case != make_case(43)
    expected = expected_database(case)
    assert score(case, expected)["passed"]
    assert not score(case, Database())["passed"]
    assert len(expected.credits) == len(expected.reports) == 2
    assert expected.reports[0].base_rate_percent == Decimal("2.15")
    assert expected.reports[1].base_rate_percent == Decimal("4.85")
    assert expected.reports[0].checking_bonus_percent == Decimal("0.65")
    assert expected.reports[0].card_bonus_percent == Decimal("0.90")
    assert expected.reports[1].checking_bonus_percent == Decimal("0.35")
    assert expected.reports[1].card_bonus_percent == Decimal("0.75")
    changed = expected.model_copy(deep=True)
    changed.credits[0].amount += Decimal("0.01")
    assert not score(case, changed)["passed"]
    expected.reports.pop()
    assert not score(case, expected)["passed"]


def test_numeric_format_and_order_are_not_task_requirements():
    case = make_case(42)
    expected = expected_database(case)
    changed = expected.model_copy(deep=True)
    changed.reports[0].checking_bonus_percent = Decimal("0.650")
    changed.credits.reverse()
    changed.reports.reverse()
    assert score(case, changed)["passed"]


@pytest.mark.parametrize(
    "balance,checking,card",
    [
        ("1234.56", "1234.56", "1234.56"),
        ("5000.00", "5000.00", "5000.00"),
        ("8833.88", "5000", "8833.88"),
        ("10000.00", "5000", "10000.00"),
        ("20511.08", "5000", "10000"),
    ],
)
def test_statement_records_include_both_policy_capped_balances(balance, checking, card):
    account = Savings(account_id="example", product="Cedar", statement_balance=balance)
    fields = account.model_dump(mode="json")
    assert Decimal(fields["checking_boosted_balance"]) == Decimal(checking)
    assert Decimal(fields["card_boosted_balance"]) == Decimal(card)
    assert set(fields) == {
        "account_id",
        "product",
        "statement_balance",
        "checking_boosted_balance",
        "card_boosted_balance",
    }


@pytest.mark.parametrize(
    "seed,checking,card,corrections",
    [
        (0, ("0.65", "0.35"), ("0.90", "0.75"), ("29.20", "91.48")),
        (1, ("0.30", "0.20"), ("0.90", "0.75"), ("19.24", "78.81")),
        (2, ("0.65", "0.35"), ("0.90", "0.40"), ("19.29", "53.73")),
    ],
)
def test_permitted_packages_use_capped_dollars_not_advertised_rates(
    seed, checking, card, corrections
):
    db = expected_database(make_case(seed))
    assert [r.checking_bonus_percent for r in db.reports] == list(map(Decimal, checking))
    assert [r.card_bonus_percent for r in db.reports] == list(map(Decimal, card))
    assert [c.amount for c in db.credits] == list(map(Decimal, corrections))


@pytest.mark.parametrize("amount", ["-1.00", "0", "1.001", "NaN", "Infinity"])
def test_invalid_credit_amount(amount):
    with pytest.raises(ValidationError):
        Credit(
            credit_id="example-credit",
            account_id="example-account",
            statement_month="2032-01",
            amount=amount,
            reason_code="example",
        )
