"""A small labeled evaluation set (~30 prompts) across four categories.

Each item has a prompt, its category, and a reference answer. The scorer
compares model output against the reference to produce a quality signal.
Kept intentionally compact so it is readable and the whole eval runs offline
in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class EvalItem:
    id: str
    category: str
    prompt: str
    reference: str


EVAL_SET: List[EvalItem] = [
    # ---------------------------- code ---------------------------------- #
    EvalItem(
        "code-01", "code",
        "Write a python function to reverse a linked list.",
        "def reverse_list(head):\n"
        "    prev = None\n"
        "    while head:\n"
        "        nxt = head.next\n"
        "        head.next = prev\n"
        "        prev = head\n"
        "        head = nxt\n"
        "    return prev",
    ),
    EvalItem(
        "code-02", "code",
        "Write a python function to check if a string is a palindrome.",
        "def is_palindrome(s):\n"
        "    s = ''.join(c.lower() for c in s if c.isalnum())\n"
        "    return s == s[::-1]",
    ),
    EvalItem(
        "code-03", "code",
        "Write a python function that returns the nth Fibonacci number iteratively.",
        "def fib(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a",
    ),
    EvalItem(
        "code-04", "code",
        "Write a python function to find the maximum value in a list without using max.",
        "def maximum(xs):\n"
        "    best = xs[0]\n"
        "    for x in xs[1:]:\n"
        "        if x > best:\n"
        "            best = x\n"
        "    return best",
    ),
    EvalItem(
        "code-05", "code",
        "Write a python function to count vowels in a string.",
        "def count_vowels(s):\n"
        "    return sum(1 for c in s.lower() if c in 'aeiou')",
    ),
    EvalItem(
        "code-06", "code",
        "Write a SQL query to select all users older than 30 from a users table.",
        "SELECT * FROM users WHERE age > 30;",
    ),
    EvalItem(
        "code-07", "code",
        "Write a python function to merge two sorted lists into one sorted list.",
        "def merge(a, b):\n"
        "    out, i, j = [], 0, 0\n"
        "    while i < len(a) and j < len(b):\n"
        "        if a[i] <= b[j]:\n"
        "            out.append(a[i]); i += 1\n"
        "        else:\n"
        "            out.append(b[j]); j += 1\n"
        "    return out + a[i:] + b[j:]",
    ),
    EvalItem(
        "code-08", "code",
        "Write a python function to remove duplicates from a list preserving order.",
        "def dedupe(xs):\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for x in xs:\n"
        "        if x not in seen:\n"
        "            seen.add(x)\n"
        "            out.append(x)\n"
        "    return out",
    ),
    # ------------------------ summarization ----------------------------- #
    EvalItem(
        "sum-01", "summarization",
        "Summarize: The mitochondria is the powerhouse of the cell, generating "
        "most of the cell's supply of ATP used as chemical energy.",
        "Mitochondria produce most of a cell's ATP energy.",
    ),
    EvalItem(
        "sum-02", "summarization",
        "Summarize: The company reported record quarterly revenue of $2 billion, "
        "up 30 percent year over year, driven by strong cloud demand.",
        "Quarterly revenue hit a record $2 billion, up 30% on strong cloud demand.",
    ),
    EvalItem(
        "sum-03", "summarization",
        "Summarize: Photosynthesis is the process by which green plants use "
        "sunlight to synthesize food from carbon dioxide and water.",
        "Plants use sunlight to make food from carbon dioxide and water.",
    ),
    EvalItem(
        "sum-04", "summarization",
        "Summarize: The treaty was signed by twelve nations after three years of "
        "negotiation, establishing new rules for maritime trade.",
        "Twelve nations signed a treaty setting new maritime trade rules.",
    ),
    EvalItem(
        "sum-05", "summarization",
        "Summarize: Regular exercise improves cardiovascular health, strengthens "
        "muscles, and reduces the risk of chronic disease.",
        "Exercise boosts heart health, builds muscle, and cuts chronic disease risk.",
    ),
    EvalItem(
        "sum-06", "summarization",
        "Summarize: The new smartphone features a faster processor, a brighter "
        "display, and a battery that lasts two days on a single charge.",
        "The new phone has a faster chip, brighter screen, and two-day battery.",
    ),
    EvalItem(
        "sum-07", "summarization",
        "Summarize: Researchers discovered a new species of frog in the rainforest "
        "that changes color to regulate its body temperature.",
        "A newly found rainforest frog changes color to regulate temperature.",
    ),
    # -------------------------- reasoning ------------------------------- #
    EvalItem(
        "reason-01", "reasoning",
        "If a train travels 60 miles in 1 hour, how far does it travel in 3 hours "
        "at the same speed? Answer with the number.",
        "180 miles",
    ),
    EvalItem(
        "reason-02", "reasoning",
        "Sarah has 5 apples and gives away 2, then buys 4 more. How many apples "
        "does she have? Answer with the number.",
        "7 apples",
    ),
    EvalItem(
        "reason-03", "reasoning",
        "If all roses are flowers and some flowers fade quickly, can we conclude "
        "all roses fade quickly? Answer yes or no with a reason.",
        "No, because only some flowers fade quickly, not necessarily roses.",
    ),
    EvalItem(
        "reason-04", "reasoning",
        "A shirt costs $20 and is discounted by 25 percent. What is the final "
        "price? Answer with the number.",
        "$15",
    ),
    EvalItem(
        "reason-05", "reasoning",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 "
        "machines to make 100 widgets? Answer with the number.",
        "5 minutes",
    ),
    EvalItem(
        "reason-06", "reasoning",
        "Two trains leave stations 300 miles apart heading toward each other at "
        "50 mph each. When do they meet? Answer with the number of hours.",
        "3 hours",
    ),
    EvalItem(
        "reason-07", "reasoning",
        "What number comes next in the sequence 2, 4, 8, 16? Answer with the number.",
        "32",
    ),
    EvalItem(
        "reason-08", "reasoning",
        "If today is Wednesday, what day will it be 10 days from now?",
        "Saturday",
    ),
    # -------------------------- extraction ------------------------------ #
    EvalItem(
        "ext-01", "extraction",
        "Extract the email from: Contact John at john.doe@example.com for details.",
        "john.doe@example.com",
    ),
    EvalItem(
        "ext-02", "extraction",
        "Extract all numbers from: I have 3 cats, 2 dogs, and 5 fish.",
        "3, 2, 5",
    ),
    EvalItem(
        "ext-03", "extraction",
        "Extract the date from: The meeting is scheduled for March 15, 2025.",
        "March 15, 2025",
    ),
    EvalItem(
        "ext-04", "extraction",
        "Extract the total from: Subtotal $40, tax $4, total $44.",
        "$44",
    ),
    EvalItem(
        "ext-05", "extraction",
        "Extract the city from: The conference will be held in Berlin next spring.",
        "Berlin",
    ),
    EvalItem(
        "ext-06", "extraction",
        "Extract the phone number from: Call us at 555-123-4567 anytime.",
        "555-123-4567",
    ),
    EvalItem(
        "ext-07", "extraction",
        "Extract the product name from: The new MacBook Pro launches in October.",
        "MacBook Pro",
    ),
]


def items_by_category(category: str) -> List[EvalItem]:
    return [item for item in EVAL_SET if item.category == category]
