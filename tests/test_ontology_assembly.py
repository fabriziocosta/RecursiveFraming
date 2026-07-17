from graphicalizer.ontology_assembly import rank_terms


def test_rank_terms_broad_level_prefers_higher_level_terms():
    terms = {
        "ROOT": {"name": "microbial entity"},
        "MID": {"name": "pathogen", "is_a": "ROOT ! microbial entity"},
        "LEAF": {"name": "viral pathogen", "is_a": "MID ! pathogen"},
    }

    broad = rank_terms(terms, {"LEAF"}, abstraction_level=2)
    specific = rank_terms(terms, {"LEAF"}, abstraction_level=0)

    assert broad.index("ROOT") < broad.index("LEAF")
    assert specific.index("LEAF") < specific.index("ROOT")


def test_rank_terms_accepts_named_abstraction_levels():
    terms = {"TERM": {"name": "pathogen"}}

    assert rank_terms(terms, {"TERM"}, abstraction_level="balanced") == ["TERM"]
