"""Invented banking cases and a strict, persistent local database."""

from __future__ import annotations

import ast
import operator
import random
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field, computed_field

Money = Annotated[Decimal, Field(ge=0, decimal_places=2, max_digits=12)]
PositiveMoney = Annotated[Decimal, Field(gt=0, decimal_places=2, max_digits=12)]
Product = Literal["Cedar", "Orchid"]


def cents(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


DOCUMENTS = {
    "monthly-interest": (
        "Monthly interest and corrections",
        (
            "This training bank uses a nominal annual rate, not an effective yield. "
            "For a full month with a constant balance, the base rate applies to the full "
            "statement balance. The checking bonus applies only to the first $5,000. "
            "The card bonus applies only to the first $10,000. "
            "Expected interest = (balance times base rate percent plus "
            "min(balance, 5000) times checking bonus percent plus "
            "min(balance, 10000) times card bonus percent) / 100 / 12. "
            "Round once, half-up to cents. Do not round the three components separately. "
            "Use the statement balance before correction credits. "
            "A package has at most one checking boost and one card boost. "
            "Use the separate product-specific tables. Current account status is not enough: "
            "all bonus qualification tests must be met during the statement month. "
            "Subtract net interest already posted for the statement month, including reversals "
            "and earlier interest corrections. Post one credit for the remaining shortfall. "
            "A credit with a different statement month is not part of the calculation."
        ),
    ),
    "checking-boosts": (
        "Checking boosts for Cedar and Orchid savings",
        (
            "Checking accounts qualify only if they are active, payroll deposits are at least "
            "$1500 for the statement month, and there are at least 10 debit purchases that month. "
            "For Cedar: Everyday gives +0.30 percentage points; Plus gives +0.65; "
            "Premier gives +1.10. For Orchid: Everyday gives +0.55; Plus gives +0.35; "
            "Premier gives +0.20. Select checking and card boosts as a permitted package. "
            "Do not sum the boosts. The best checking account can differ by savings product."
        ),
    ),
    "card-boosts": (
        "Card boosts for Cedar and Orchid savings",
        (
            "Cards qualify only if they are open, are not delinquent, and statement-month "
            "qualifying purchases are at least $500. For Cedar: Rewards gives +0.35 "
            "percentage points; Platinum gives +0.60; Travel gives +0.90. "
            "For Orchid: Rewards gives +0.75; Platinum gives +0.25; Travel gives +0.40. "
            "Use only an eligible card boost in a permitted relationship package. "
            "Do not sum the boosts. Do not use a card's advertised boost without qualification."
        ),
    ),
    "relationship-packages": (
        "Relationship package eligibility and automatic pricing",
        (
            "The bank automatically applies the permitted package that pays the most interest "
            "dollars for each savings account. No customer enrollment is needed. "
            "Compare min(balance, 5000) times checking bonus percent plus "
            "min(balance, 10000) times card bonus percent. "
            "Do not select a package by the sum of its advertised percentages. "
            "If two packages pay the same amount, prefer the higher card bonus, then the "
            "higher checking bonus. "
            "A package may use one eligible checking boost, one eligible card boost, or both. "
            "A package may leave either category unused. The following exclusions apply: "
            "Cedar cannot combine a Premier checking boost with a Travel card boost. "
            "Orchid cannot combine an Everyday checking boost with a Rewards card boost. "
            "Each excluded combination may still supply one of its two boosts on its own. "
            "All other checking and card combinations are permitted. "
            "One relationship account may support both savings accounts. "
            "The package rule overrides any general direction to select each category separately."
        ),
    ),
    "rates-and-reports": (
        "Base rates and discrepancy reports",
        (
            "Cedar has a 2.15 percent nominal annual base rate. Orchid has a 4.85 percent "
            "nominal annual base rate. After posting a correction credit, save one discrepancy "
            "report per affected savings account. The report contains the statement month, "
            "the base rate percent, checking bonus percent, card bonus percent, "
            "checking boosted balance, card boosted balance, "
            "expected interest, net interest already posted before this run, and the correction "
            "credit ID. Use reason_code interest_shortfall "
            "for correction credits. Do not change an unrelated account or month."
        ),
    ),
}


class Savings(BaseModel):
    account_id: str
    product: Product
    statement_balance: Money

    @computed_field
    @property
    def checking_boosted_balance(self) -> Decimal:
        return min(self.statement_balance, Decimal(5000))

    @computed_field
    @property
    def card_boosted_balance(self) -> Decimal:
        return min(self.statement_balance, Decimal(10000))


class Checking(BaseModel):
    account_id: str
    product: Literal["Everyday", "Plus", "Premier"]
    active: bool
    payroll: Money
    debit_purchases: int = Field(ge=0)


class Card(BaseModel):
    card_id: str
    product: Literal["Rewards", "Platinum", "Travel"]
    open: bool
    delinquent: bool
    purchases: Money


class InterestEntry(BaseModel):
    account_id: str
    statement_month: str
    kind: Literal["interest", "reversal", "correction"]
    amount: Decimal


class Case(BaseModel):
    case_id: str
    statement_month: str = "2031-04"
    savings: list[Savings]
    checking: list[Checking]
    cards: list[Card]
    interest_entries: list[InterestEntry]


class Credit(BaseModel):
    credit_id: str
    account_id: str
    statement_month: str
    amount: PositiveMoney
    reason_code: str


class InterestCalculation(BaseModel):
    account_id: str
    statement_month: str
    base_rate_percent: Annotated[Decimal, Field(ge=0, le=100)]
    checking_bonus_percent: Annotated[Decimal, Field(ge=0, le=100)]
    card_bonus_percent: Annotated[Decimal, Field(ge=0, le=100)]
    checking_boosted_balance: Money
    card_boosted_balance: Money
    expected_interest: Money
    net_interest_before_run: Decimal


class Report(InterestCalculation):
    correction_credit_id: str


class PackageQuote(InterestCalculation):
    permitted: Literal[True] = True
    checking_account_id: str | None
    card_id: str | None
    annual_bonus: Decimal
    correction_amount: Decimal


class QuoteRefusal(BaseModel):
    permitted: Literal[False] = False
    reason: str


class Database(BaseModel):
    credits: list[Credit] = Field(default_factory=list)
    reports: list[Report] = Field(default_factory=list)


def make_case(seed: int) -> Case:
    rng = random.Random(seed)
    prefix = f"case-{seed:06d}"
    savings = [
        Savings(
            account_id=f"{prefix}-s{index}",
            product=product,
            statement_balance=(
                Decimal(rng.randrange(700000, 2400000)) / 100
                + (Decimal(5000) if seed % 3 == 2 and product == "Orchid" else 0)
            ),
        )
        for index, product in enumerate(("Cedar", "Orchid"), 1)
    ]
    entries = []
    for account in savings:
        account_entries: list[
            tuple[str, Literal["interest", "reversal", "correction"], Decimal]
        ] = [
            ("2031-04", "interest", Decimal(rng.randrange(500, 1000)) / 100),
            ("2031-04", "reversal", -Decimal(rng.randrange(100, 300)) / 100),
            ("2031-04", "correction", Decimal(rng.randrange(50, 150)) / 100),
            ("2031-03", "interest", Decimal(rng.randrange(1200, 2000)) / 100),
        ]
        for month, kind, amount in account_entries:
            entries.append(
                InterestEntry(
                    account_id=account.account_id,
                    statement_month=month,
                    kind=kind,
                    amount=amount,
                )
            )
    rng.shuffle(entries)
    return Case(
        case_id=prefix,
        savings=savings,
        checking=[
            Checking(
                account_id=f"{prefix}-c1",
                product="Everyday",
                active=True,
                payroll=Decimal(1200 if seed % 3 == 2 else 2100),
                debit_purchases=14,
            ),
            Checking(
                account_id=f"{prefix}-c2",
                product="Plus",
                active=True,
                payroll=Decimal(1800),
                debit_purchases=8 if seed % 3 == 1 else 12,
            ),
            Checking(
                account_id=f"{prefix}-c3",
                product="Premier",
                active=True,
                payroll=Decimal(2600),
                debit_purchases=16,
            ),
        ],
        cards=[
            Card(
                card_id=f"{prefix}-k1",
                product="Rewards",
                open=True,
                delinquent=False,
                purchases=Decimal(200 if seed % 3 == 2 else 800),
            ),
            Card(
                card_id=f"{prefix}-k2",
                product="Platinum",
                open=True,
                delinquent=False,
                purchases=Decimal(200 if seed % 3 == 1 else 1200),
            ),
            Card(
                card_id=f"{prefix}-k3",
                product="Travel",
                open=True,
                delinquent=False,
                purchases=Decimal(900),
            ),
        ],
        interest_entries=entries,
    )


def quote_package(
    case: Case,
    account_id: str,
    checking_account_id: str | None = None,
    card_id: str | None = None,
) -> PackageQuote | QuoteRefusal:
    account = next((a for a in case.savings if a.account_id == account_id), None)
    if account is None:
        raise ValueError("The savings account is not in this case.")
    checking = next((a for a in case.checking if a.account_id == checking_account_id), None)
    card = next((a for a in case.cards if a.card_id == card_id), None)
    if checking_account_id is not None and (
        checking is None
        or not checking.active
        or checking.payroll < 1500
        or checking.debit_purchases < 10
    ):
        return QuoteRefusal(reason="The checking account does not qualify for this statement.")
    if card_id is not None and (
        card is None or not card.open or card.delinquent or card.purchases < 500
    ):
        return QuoteRefusal(reason="The card does not qualify for this statement.")
    pair = (checking.product if checking else None, card.product if card else None)
    if pair == {"Cedar": ("Premier", "Travel"), "Orchid": ("Everyday", "Rewards")}[account.product]:
        return QuoteRefusal(reason="The relationship policy does not permit this package.")
    checking_rates = {
        "Cedar": {"Everyday": "0.30", "Plus": "0.65", "Premier": "1.10"},
        "Orchid": {"Everyday": "0.55", "Plus": "0.35", "Premier": "0.20"},
    }
    card_rates = {
        "Cedar": {"Rewards": "0.35", "Platinum": "0.60", "Travel": "0.90"},
        "Orchid": {"Rewards": "0.75", "Platinum": "0.25", "Travel": "0.40"},
    }
    base = Decimal("2.15" if account.product == "Cedar" else "4.85")
    checking_bonus = (
        Decimal(checking_rates[account.product][checking.product]) if checking else Decimal(0)
    )
    card_bonus = Decimal(card_rates[account.product][card.product]) if card else Decimal(0)
    bonus = (
        account.checking_boosted_balance * checking_bonus
        + account.card_boosted_balance * card_bonus
    ) / 100
    expected = cents((account.statement_balance * base / 100 + bonus) / 12)
    posted = sum(
        (
            entry.amount
            for entry in case.interest_entries
            if entry.account_id == account_id and entry.statement_month == case.statement_month
        ),
        Decimal(0),
    )
    return PackageQuote(
        account_id=account_id,
        statement_month=case.statement_month,
        base_rate_percent=base,
        checking_bonus_percent=checking_bonus,
        card_bonus_percent=card_bonus,
        checking_boosted_balance=account.checking_boosted_balance,
        card_boosted_balance=account.card_boosted_balance,
        expected_interest=expected,
        net_interest_before_run=posted,
        checking_account_id=checking_account_id,
        card_id=card_id,
        annual_bonus=bonus,
        correction_amount=expected - posted,
    )


def expected_database(case: Case) -> Database:
    """Independent scorer; this function is never registered as an agent tool."""
    checking = {
        "Cedar": {"Everyday": "0.30", "Plus": "0.65", "Premier": "1.10"},
        "Orchid": {"Everyday": "0.55", "Plus": "0.35", "Premier": "0.20"},
    }
    cards = {
        "Cedar": {"Rewards": "0.35", "Platinum": "0.60", "Travel": "0.90"},
        "Orchid": {"Rewards": "0.75", "Platinum": "0.25", "Travel": "0.40"},
    }
    bases = {"Cedar": Decimal("2.15"), "Orchid": Decimal("4.85")}
    db = Database()
    for account in case.savings:
        eligible_checking = [None] + [
            item
            for item in case.checking
            if item.active and item.payroll >= 1500 and item.debit_purchases >= 10
        ]
        eligible_cards = [None] + [
            item
            for item in case.cards
            if item.open and not item.delinquent and item.purchases >= 500
        ]
        excluded = {"Cedar": ("Premier", "Travel"), "Orchid": ("Everyday", "Rewards")}
        packages = []
        checking_balance = min(account.statement_balance, Decimal(5000))
        card_balance = min(account.statement_balance, Decimal(10000))
        for bank_account in eligible_checking:
            for card in eligible_cards:
                pair = (
                    bank_account.product if bank_account else None,
                    card.product if card else None,
                )
                if pair == excluded[account.product]:
                    continue
                checking_bonus = (
                    Decimal(checking[account.product][bank_account.product])
                    if bank_account
                    else Decimal(0)
                )
                card_bonus = Decimal(cards[account.product][card.product]) if card else Decimal(0)
                weighted_bonus = checking_balance * checking_bonus + card_balance * card_bonus
                packages.append((weighted_bonus, card_bonus, checking_bonus))
        weighted_bonus, card_bonus, checking_bonus = max(packages)
        expected = cents(
            (account.statement_balance * bases[account.product] + weighted_bonus) / 100 / 12
        )
        posted = sum(
            (
                entry.amount
                for entry in case.interest_entries
                if entry.account_id == account.account_id
                and entry.statement_month == case.statement_month
            ),
            Decimal(0),
        )
        correction = expected - posted
        if correction <= 0:
            continue
        identifier = f"credit-{account.account_id}-{case.statement_month}"
        db.credits.append(
            Credit(
                credit_id=identifier,
                account_id=account.account_id,
                statement_month=case.statement_month,
                amount=correction,
                reason_code="interest_shortfall",
            )
        )
        db.reports.append(
            Report(
                account_id=account.account_id,
                statement_month=case.statement_month,
                base_rate_percent=bases[account.product],
                checking_bonus_percent=checking_bonus,
                card_bonus_percent=card_bonus,
                checking_boosted_balance=checking_balance,
                card_boosted_balance=card_balance,
                expected_interest=expected,
                net_interest_before_run=posted,
                correction_credit_id=identifier,
            )
        )
    return db


def score(case: Case, actual: Database) -> dict:
    expected = expected_database(case)

    # The order of independent account operations is not part of the task.
    def normalized(db: Database) -> dict:
        return {
            key: sorted(value, key=lambda item: item["account_id"])
            for key, value in db.model_dump().items()
        }

    wanted, got = normalized(expected), normalized(actual)
    return {
        "passed": wanted == got,
        "expected": wanted,
        "actual": got,
        "differences": {
            key: {"expected": wanted[key], "actual": got[key]}
            for key in wanted
            if wanted[key] != got[key]
        },
    }


def calculate(expression: str) -> str:
    """Evaluate arithmetic and round half-up to two decimal places."""
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return Decimal(ast.get_source_segment(expression, node) or "")
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        raise ValueError(
            "Only numeric addition, subtraction, multiplication and division are allowed."
        )

    if len(expression) > 500:
        raise ValueError("The expression is too long.")
    result = visit(ast.parse(expression, mode="eval").body)
    if not result.is_finite():
        raise ValueError("The result must be finite.")
    return str(cents(result))
