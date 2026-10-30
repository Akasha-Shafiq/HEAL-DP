"""
    Approved query registry for the MIMIC-DP platform 
    
    The coordinatore accepts only queries defined in this registry. 
    Each query specifies its statistical semantics and the metadata required 
    by the distrubuted and privacy-preserving workflow.
"""

from copy import deepcopy
# ---------------------------------------------------------------
##################Global experiment policy########################
# ---------------------------------------------------------------

PRIVACY_UNIT = "subject_id"
ALLOWED_EPSILONS = (
    0.1, 0.5, 1.0, 5.0, 10.0    
)

# ---------------------------------------------------------------
##################Aproved query definitions########################
# ---------------------------------------------------------------

QUERY_REGISTRY = {
    "Q001":{
        "query_id": "Q001",
        "name": "Unique patient count",
        
        "description": (
            "Count the number of unqiue patients across "
            "the participating hospitals."
        ),
        
        "privacy_unit": PRIVACY_UNIT,
        
        "contribution_rule":(
            "Each patient contributes once."
            ),
        
        "query_type": "count",
        
        "aggregation": "sum",
        
        "bounds":None,
        
        "allowed_epsilons": ALLOWED_EPSILONS,
        
        "enabled": True,
        
    },
    
    "Q002": {
        "query_id": "Q002",
        "name": "Admission-type histogram",

        "description": (
            "Patient-level histogram of admission_type "
            "using each patient's earliest admission."
        ),

        "privacy_unit": PRIVACY_UNIT,

        "contribution_rule": (
            "Each patient contributes to exactly one "
            "admission_type category using the earliest admission."
        ),

        "query_type": "histogram",

        "aggregation": "elementwise_sum",

        "bounds": None,

        "allowed_epsilons": ALLOWED_EPSILONS,

        "enabled": True,
    },

    "Q003": {
        "query_id": "Q003",
        "name": "Bounded mean length of stay",

        "description": (
        "Mean hospital length of stay using the patient's "
        "earliest admission and LOS clipped to [0, 30] days."
    ),

        "privacy_unit": PRIVACY_UNIT,

        "contribution_rule": (
            "Each patient contributes one bounded LOS value "
            "from the earliest admission."
        ),

        "query_type": "bounded_mean",

        "aggregation": "sum_and_count",

        "bounds": {
            "lower": 0.0,
            "upper": 30.0,
            "unit": "days",
        },

        "allowed_epsilons": ALLOWED_EPSILONS,

        "enabled": True,
    },

    "Q004": {
        "query_id": "Q004",
        "name": "Race by admission-type contingency table",

        "description": (
            "Two-dimensional patient-level histogram of race "
            "and admission_type using the earliest admission."
        ),

        "privacy_unit": PRIVACY_UNIT,

        "contribution_rule": (
            "Each patient contributes to exactly one joint "
            "(race, admission_type) cell."
        ),

        "query_type": "joint_histogram",

        "aggregation": "elementwise_sum",

        "bounds": None,

        "allowed_epsilons": ALLOWED_EPSILONS,

        "enabled": True,
    },

    "Q005": {
        "query_id": "Q005",
        "name": "Bounded mean LOS by race",

        "description": (
            "Patient-level bounded mean length of stay "
            "calculated separately for each race group."
        ),

        "privacy_unit": PRIVACY_UNIT,

        "contribution_rule": (
            "Each patient contributes one bounded LOS value "
            "to exactly one race group."
        ),

        "query_type": "grouped_bounded_mean",

        "aggregation": "grouped_sum_and_count",

        "bounds": {
            "lower": 0.0,
            "upper": 30.0,
            "unit": "days",
        },

        "allowed_epsilons": ALLOWED_EPSILONS,

        "enabled": True,
    },
    
    
    
    "Q006": {
    "query_id": "Q006",
    "name": "Rare subgroup count: ASIAN - KOREAN and URGENT",
    "description": (
        "Count patients whose earliest admission has race "
        "'ASIAN - KOREAN' and admission type 'URGENT'."
    ),
    "query_type": "count",
    "privacy_unit": PRIVACY_UNIT,
    "aggregation": "sum",
    "allowed_epsilons": ALLOWED_EPSILONS,
    "enabled": True,
    "predicate": {
        "race": "ASIAN - KOREAN",
        "admission_type": "URGENT",
    },
},
    
    
    "Q007": {
    "query_id": "Q007",
    "name": "Rare subgroup bounded mean length of stay",

    "description": (
        "Mean hospital length of stay for patients whose earliest "
        "admission has race 'ASIAN - KOREAN' and admission type "
        "'URGENT', with LOS clipped to [0, 30] days."
    ),

    "privacy_unit": PRIVACY_UNIT,

    "contribution_rule": (
        "Each matching patient contributes one bounded LOS value "
        "from the earliest admission."
    ),

    "query_type": "bounded_mean",

    "aggregation": "sum_and_count",

    "bounds": {
        "lower": 0.0,
        "upper": 30.0,
        "unit": "days",
    },

    "predicate": {
        "race": "ASIAN - KOREAN",
        "admission_type": "URGENT",
    },

    "allowed_epsilons": ALLOWED_EPSILONS,

    "enabled": True,
},
    
        
}


# ------------------------------------------------------------------------
########################Resgistry access functions########################
# ------------------------------------------------------------------------

def get_query(query_id: str) -> dict:
    
    """Return the approved definition for a query 
    
    Raises 
    key error 
    if the query is not registered or is disabled.
    """
    
    query_id = query_id.upper()
    if query_id not in QUERY_REGISTRY:
        raise KeyError(f"Query {query_id} is not registered.")
    
    query= QUERY_REGISTRY[query_id]
    
    if not query["enabled"]:
        raise KeyError(
            f"Query '{query_id}' is not registered."
        )

    return deepcopy(query)


def list_queries() -> list[dict]:
    """
    Return all currently enabled approved queries.
    """

    return [
        deepcopy(query)
        for query in QUERY_REGISTRY.values()
        if query["enabled"]
    ]
    