"""
Object-Oriented Design Interview Questions

This module contains challenging OOD questions for mock interviews.
Questions are designed to be rigorous and test deep understanding of:
- SOLID principles
- Design patterns
- Object-oriented concepts
- System scalability
- Real-world constraints
"""

import random
from typing import Dict, List, Optional

STOCK_MATCH_ENGINE_QUESTION: Dict[str, str] = {
    "title": "Stock Exchange Matching Engine",
    "description": """Design a high-performance stock exchange order matching engine.\n\nRequirements:\n- Maintain a live limit order book for thousands of symbols\n- Support limit, market, stop, iceberg, and cancel/replace orders\n- Match orders using price-time priority; support partial fills\n- Persist trade history and current book snapshots\n- Handle millions of orders per second with low latency\n- Provide real-time market data feeds (top of book + depth)\n- Enforce risk controls and per-account limits\n\nDig deep into:\n- Core data structures for bids/asks and fast lookups\n- Matching algorithm flow and edge cases (e.g., self-trade prevention)\n- Concurrency, sharding, and recovery strategy\n- How you'd test and monitor correctness/latency""",
    "difficulty": "Hard",
    "keywords": ["stock", "exchange", "match", "trading", "market", "order", "latency", "fintech"],
}


OOD_QUESTIONS: List[Dict[str, str]] = [
    STOCK_MATCH_ENGINE_QUESTION,
    {
        "title": "Parking Lot System",
        "description": """Design a parking lot management system.

Requirements:
- Multiple levels with different spot sizes (compact, regular, large)
- Track occupied and available spots
- Support different vehicle types (motorcycle, car, bus)
- Calculate parking fees based on duration
- Handle concurrent access

Think about: What classes do you need? How do vehicles find spots? How do you handle the bus needing multiple spots?""",
        "difficulty": "Medium-Hard",
        "keywords": ["parking", "mobility", "transport", "rideshare", "uber", "lyft", "vehicle", "garage", "autonomous"]
    },
    {
        "title": "Elevator Control System",
        "description": """Design an elevator control system for a building.

Requirements:
- Multiple elevators serving multiple floors
- Optimize for minimal wait time
- Handle both up and down requests
- Support emergency stops and maintenance mode
- Different priority levels (VIP vs regular)

Think about: What's your scheduling algorithm? How do elevators communicate? How do you prevent starvation?""",
        "difficulty": "Hard",
        "keywords": ["elevator", "hardware", "robotics", "industrial", "automation", "iot", "building", "manufacturing"]
    },
    {
        "title": "Library Management System",
        "description": """Design a library management system.

Requirements:
- Track books, members, checkouts, and reservations
- Support multiple copies of the same book
- Handle late fees
- Reservation queue when book is unavailable
- Different member types with different checkout limits

Think about: How do you model book vs book copy? How do you handle the reservation queue? What happens when a reserved book is returned?""",
        "difficulty": "Medium",
        "keywords": ["library", "education", "books", "publishing", "edtech", "university", "school"]
    },
    {
        "title": "Chess Game",
        "description": """Design a chess game.

Requirements:
- Two players alternate turns
- Each piece type has unique movement rules
- Validate legal moves
- Detect check, checkmate, and stalemate
- Support special moves (castling, en passant, pawn promotion)

Think about: How do you represent the board? How do you validate moves? How do you detect check? Should Piece be abstract?""",
        "difficulty": "Hard",
        "keywords": ["chess", "gaming", "game", "strategy", "puzzle", "board"]
    },
    {
        "title": "Hotel Reservation System",
        "description": """Design a hotel reservation system.

Requirements:
- Multiple room types with different rates
- Search availability by date range
- Handle bookings, modifications, cancellations
- Support room upgrades
- Track room status (available, occupied, maintenance)

Think about: How do you handle overlapping reservations? How do you search efficiently? What about partial cancellations?""",
        "difficulty": "Medium-Hard",
        "keywords": ["hotel", "hospitality", "travel", "airbnb", "expedia", "booking", "resort"]
    },
    {
        "title": "ATM Machine",
        "description": """Design an ATM machine system.

Requirements:
- Support withdrawals, deposits, balance inquiry
- Handle multiple accounts per card
- Validate PIN with limited attempts
- Dispense correct bills (minimize number of bills)
- Handle insufficient funds and out-of-cash scenarios

Think about: How do you model cash dispensing? How do you handle transactions atomically? What happens if the machine runs out of a denomination?""",
        "difficulty": "Medium-Hard",
        "keywords": ["bank", "fintech", "finance", "payments", "atm", "stripe", "paypal", "square"]
    },
    {
        "title": "Deck of Cards",
        "description": """Design a generic deck of cards that can be used for different card games.

Requirements:
- Support standard 52-card deck and variations
- Shuffle operation
- Deal cards to players
- Reset and reshuffle
- Support different games (Poker, Blackjack, etc.)

Think about: Is Card mutable or immutable? How do you handle suit and rank? How do you make it extensible for different games?""",
        "difficulty": "Medium",
        "keywords": ["cards", "casino", "gambling", "poker", "blackjack", "gaming"]
    },
    {
        "title": "Restaurant Reservation System",
        "description": """Design a restaurant reservation system.

Requirements:
- Manage tables of different sizes
- Handle walk-ins and reservations
- Support party size matching to table
- Time-slot management
- Waitlist when fully booked

Think about: How do you assign parties to tables? How do you handle the waitlist? What happens when a party is late or doesn't show up?""",
        "difficulty": "Medium-Hard",
        "keywords": ["restaurant", "food", "dining", "hospitality", "diner", "ubereats", "doordash", "grubhub"]
    },
    {
        "title": "File System",
        "description": """Design an in-memory file system.

Requirements:
- Support directories and files in a tree structure
- Create, delete, move, copy operations
- Search by name or extension
- Calculate directory size recursively
- Support file permissions

Think about: How do you model file vs directory? What design pattern fits here? How do you handle circular references with symlinks?""",
        "difficulty": "Hard",
        "keywords": ["filesystem", "storage", "os", "cloud", "infrastructure", "dropbox", "box", "sre", "kernel", "aws", "azure", "gcp"]
    },
    {
        "title": "Online Shopping Cart",
        "description": """Design an online shopping cart and checkout system.

Requirements:
- Add/remove items, update quantities
- Apply discount codes and promotions
- Calculate taxes based on location
- Support multiple payment methods
- Handle inventory validation before checkout

Think about: How do you handle promotions (buy one get one, percentage off)? How do you ensure items are still available at checkout? What if prices change?""",
        "difficulty": "Medium-Hard",
        "keywords": ["ecommerce", "shopping", "retail", "amazon", "marketplace", "shopify", "checkout"]
    },
    {
        "title": "Movie Ticket Booking System",
        "description": """Design a movie ticket booking system.

Requirements:
- Multiple theaters with multiple screens
- Different showtimes and movies
- Seat selection with real-time availability
- Hold seats temporarily during booking
- Support group bookings

Think about: How do you handle concurrent bookings for the same seat? How long do you hold seats? How do you represent the seat layout?""",
        "difficulty": "Hard",
        "keywords": ["movie", "cinema", "ticket", "entertainment", "netflix", "disney", "hbo", "theater", "fandango"]
    },
    {
        "title": "Social Media Feed",
        "description": """Design a social media news feed system.

Requirements:
- Users create posts (text, images, videos)
- Follow/unfollow other users
- Generate personalized feed
- Like, comment, share functionality
- Different post types with different rendering

Think about: How do you model different post types? How do you generate the feed efficiently? How do you handle privacy settings?""",
        "difficulty": "Hard",
        "keywords": ["social", "media", "feed", "content", "facebook", "instagram", "twitter", "snap", "tiktok", "linkedin", "youtube"]
    },
]


def get_random_question() -> Dict[str, str]:
    """Get a random OOD question"""
    return random.choice(OOD_QUESTIONS)


def get_stock_match_engine_question() -> Dict[str, str]:
    """Always return the stock exchange matching engine question."""
    return dict(STOCK_MATCH_ENGINE_QUESTION)


def get_question_by_difficulty(difficulty: str) -> Dict[str, str]:
    """Get a random question filtered by difficulty"""
    filtered = [q for q in OOD_QUESTIONS if q["difficulty"] == difficulty]
    if not filtered:
        return get_random_question()
    return random.choice(filtered)


def _context_to_text(company_context: Optional[Dict[str, str]]) -> str:
    if not company_context:
        return ""
    parts = [company_context.get("company", ""), company_context.get("role", ""), company_context.get("details", "")]
    return " ".join(part for part in parts if part).lower()


def _score_question(question: Dict[str, str], context_text: str) -> int:
    if not context_text:
        return 0
    score = 0
    for keyword in question.get("keywords", []):
        keyword_lower = keyword.lower()
        if keyword_lower and keyword_lower in context_text:
            score += 2 if len(keyword_lower) >= 6 else 1
    return score


def get_hard_question(company_context: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Get a challenging question, biased toward the company/role context when provided."""
    filtered = [q for q in OOD_QUESTIONS if q["difficulty"] in ["Hard", "Medium-Hard"]]
    if not filtered:
        return get_random_question()

    context_text = _context_to_text(company_context)
    if not context_text:
        return random.choice(filtered)

    best_question: Optional[Dict[str, str]] = None
    best_score = 0
    for question in filtered:
        score = _score_question(question, context_text)
        if score > best_score:
            best_score = score
            best_question = question

    if best_question and best_score > 0:
        return best_question

    return random.choice(filtered)
