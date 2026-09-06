"""
content.py
A small bank of sample passages for the demo. In a real system this
would be a much larger, curriculum-aligned content library. Each passage
is pre-split into sentences so the adaptive engine can chunk it dynamically.
"""

PASSAGES = [
    {
        "title": "The Curious Tortoise",
        "grade_levels": ["Beginner Reader", "Developing Reader"],
        "sentences": [
            "Tomi the tortoise loved to explore the garden every morning.",
            "One day, she found a shiny red ball under the mango tree.",
            "She pushed the ball with her nose, and it rolled down the hill.",
            "Tomi chased after it slowly, but she did not give up.",
            "At the bottom of the hill, she found the ball waiting for her.",
            "Tomi smiled and rolled the ball all the way back home.",
        ],
        "quiz": [
            {
                "q": "What did Tomi find under the mango tree?",
                "options": ["A shiny red ball", "A book", "A shoe"],
                "answer": "A shiny red ball",
            },
            {
                "q": "What did Tomi do when the ball rolled down the hill?",
                "options": ["She cried", "She chased it slowly", "She went home"],
                "answer": "She chased it slowly",
            },
            {
                "q": "How did the story end?",
                "options": [
                    "Tomi lost the ball",
                    "Tomi rolled the ball back home",
                    "Tomi fell asleep",
                ],
                "answer": "Tomi rolled the ball back home",
            },
        ],
    },
    {
        "title": "Amaka's Market Day",
        "grade_levels": ["Developing Reader", "Fluent Reader"],
        "sentences": [
            "Every Saturday, Amaka went to the market with her mother.",
            "The market was full of colorful fruits, baskets, and noise.",
            "Amaka's favorite stall sold sweet oranges and groundnuts.",
            "This week, she helped her mother count change for a customer.",
            "The seller smiled and gave Amaka a free orange for helping.",
            "On the walk home, Amaka told her mother she wanted to be a trader too.",
        ],
        "quiz": [
            {
                "q": "What day did Amaka go to the market?",
                "options": ["Saturday", "Monday", "Sunday"],
                "answer": "Saturday",
            },
            {
                "q": "What did Amaka help her mother do?",
                "options": ["Cook food", "Count change", "Carry water"],
                "answer": "Count change",
            },
            {
                "q": "What did the seller give Amaka?",
                "options": ["A free orange", "Some money", "A basket"],
                "answer": "A free orange",
            },
        ],
    },
    {
        "title": "The Fastest Bicycle",
        "grade_levels": ["Fluent Reader"],
        "sentences": [
            "Chuka had been saving his allowance for three months to buy a bicycle.",
            "Finally, his father took him to the market to choose one.",
            "Chuka picked a blue bicycle with a bell and a basket in front.",
            "On his first ride, he raced his friend Musa down the street.",
            "Chuka pedaled as fast as he could, and his heart beat quickly with excitement.",
            "He crossed the finish line first and raised his hands in victory.",
        ],
        "quiz": [
            {
                "q": "How long did Chuka save for the bicycle?",
                "options": ["Three months", "One week", "A year"],
                "answer": "Three months",
            },
            {
                "q": "What color was the bicycle Chuka picked?",
                "options": ["Red", "Blue", "Green"],
                "answer": "Blue",
            },
            {
                "q": "Who did Chuka race?",
                "options": ["His father", "Musa", "His teacher"],
                "answer": "Musa",
            },
        ],
    },
]


def get_passages_for_grade(reading_level):
    """
    NOTE: 'reading_level' selects appropriately-leveled content ONLY.
    It is never used for ADHD screening or classification -- age and
    the behavioral/cognitive screening tasks handle that (see ml_engine.py).
    """
    matches = [p for p in PASSAGES if reading_level in p["grade_levels"]]
    return matches if matches else PASSAGES
