
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Dubai Rental Decision Support",
    page_icon="🏙️",
    layout="wide"
)


# ==================================================
# LOAD FROZEN MODEL
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR

model = joblib.load(
    ARTIFACT_DIR /
    "dubai_rent_xgboost.joblib"
)

preprocessor = joblib.load(
    ARTIFACT_DIR /
    "dubai_rent_preprocessor.joblib"
)

with open(
    ARTIFACT_DIR /
    "model_metadata.json",
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


with open(
    ARTIFACT_DIR /
    "portal_categories.json",
    "r",
    encoding="utf-8"
) as f:

    categories = json.load(f)


support_data = pd.read_csv(
    ARTIFACT_DIR /
    "area_support.csv"
)


# ==================================================
# PROPERTY CONFIGURATION MAP
# ==================================================

config_map = {}

for config in categories[
    "property_configs"
]:

    parts = config.split(
        " | ",
        1
    )

    if len(parts) == 2:
        property_type = parts[0]
        configuration = parts[1]

    else:
        property_type = config
        configuration = config

    if property_type not in config_map:
        config_map[property_type] = []

    config_map[property_type].append(
        configuration
    )


for property_type in config_map:

    config_map[property_type] = sorted(
        list(
            set(
                config_map[property_type]
            )
        )
    )


month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}


# ==================================================
# PREDICTION FUNCTION
# ==================================================

def predict_rent(
    property_size,
    area_name,
    property_config,
    contract_type,
    contract_year,
    contract_month,
    quoted_rent=None
):

    input_data = pd.DataFrame({

        "actual_area_clean": [
            float(property_size)
        ],

        "contract_year": [
            int(contract_year)
        ],

        "area_name_en": [
            area_name
        ],

        "property_config": [
            property_config
        ],

        "contract_month": [
            int(contract_month)
        ],

        "contract_reg_type_en": [
            contract_type
        ]
    })


    encoded = preprocessor.transform(
        input_data
    )


    predicted = float(
        model.predict(
            encoded
        )[0]
    )


    half_width = float(
        metadata[
            "prediction_range_half_width_AED"
        ]
    )


    raw_lower = (
        predicted
        - half_width
    )

    raw_upper = (
        predicted
        + half_width
    )


    result = {

        "prediction":
            predicted,

        "lower":
            max(
                0,
                raw_lower
            ),

        "upper":
            raw_upper
    }


    if quoted_rent is not None:

        quoted_rent = float(
            quoted_rent
        )

        difference = (
            quoted_rent
            - predicted
        )

        difference_percent = (
            100
            * difference
            / predicted
        )


        if quoted_rent < raw_lower:

            position = (
                "Below predicted range"
            )

        elif quoted_rent > raw_upper:

            position = (
                "Above predicted range"
            )

        else:

            position = (
                "Within predicted range"
            )


        result.update({

            "quoted_rent":
                quoted_rent,

            "difference":
                difference,

            "difference_percent":
                difference_percent,

            "position":
                position
        })


    return result


# ==================================================
# SIMPLE ROUTING
# ==================================================

if "page" not in st.session_state:

    st.session_state.page = (
        "home"
    )


def go_to(page):

    st.session_state.page = page

    st.rerun()


# ==================================================
# SHARED BACK BUTTON
# ==================================================

def back_button():

    if st.button(
        "← Back to tools",
        key="back_home"
    ):

        go_to(
            "home"
        )


# ==================================================
# HOME PAGE
# ==================================================

if st.session_state.page == "home":


    # ----------------------------------------------
    # HERO
    # ----------------------------------------------

    st.title(
        "Dubai Rental Decision Support"
    )

    st.write(
        """
        Use historical Dubai rental-contract data
        to explore rental estimates, compare quotes,
        search areas by budget and compare the
        financial cost of staying versus moving.
        """
    )

    st.divider()

    st.subheader(
        "What would you like to do?"
    )


    # ----------------------------------------------
    # THREE TOOL CARDS
    # ----------------------------------------------

    col1, col2, col3 = st.columns(
        3,
        gap="large"
    )


    with col1:

        st.markdown(
            "### 🏠 Rent Estimate"
        )

        st.write(
            """
            Estimate the annual rent for a property,
            view its prediction range and compare a
            rental quote with the model estimate.
            """
        )

        if st.button(
            "Open Rent Estimate",
            use_container_width=True,
            type="primary",
            key="home_estimator"
        ):

            go_to(
                "estimate"
            )


    with col2:

        st.markdown(
            "### 📍 Budget Explorer"
        )

        st.write(
            """
            Enter your budget and property
            requirements to explore Dubai areas
            that may fit your selected budget.
            """
        )

        if st.button(
            "Open Budget Explorer",
            use_container_width=True,
            type="primary",
            key="home_budget"
        ):

            go_to(
                "budget"
            )


    with col3:

        st.markdown(
            "### 🔄 Stay vs Move"
        )

        st.write(
            """
            Compare the financial cost of renewing
            your current property with moving to an
            alternative property over several years.
            """
        )

        if st.button(
            "Open Stay vs Move",
            use_container_width=True,
            type="primary",
            key="home_move"
        ):

            go_to(
                "move"
            )


    st.divider()

    st.caption(
        """
        The portal is a research prototype and is
        intended for decision support rather than
        official rental valuation.
        """
    )


# ==================================================
# RENT ESTIMATE PAGE
# ==================================================

elif st.session_state.page == "estimate":


    back_button()

    st.title(
        "🏠 Rent Estimate"
    )

    st.write(
        """
        Enter the property details to estimate the
        annual rent and compare an optional rental
        quote.
        """
    )

    st.divider()


    left, right = st.columns(
        [
            1,
            1
        ],
        gap="large"
    )


    with left:

        st.subheader(
            "Property details"
        )


        area_name = st.selectbox(
            "Area",
            categories[
                "areas"
            ],
            key="estimate_area"
        )


        property_type = st.selectbox(
            "Property type",
            sorted(
                config_map.keys()
            ),
            key="estimate_type"
        )


        configuration = st.selectbox(
            "Bedrooms / configuration",
            config_map[
                property_type
            ],
            key="estimate_config"
        )


        property_config = (
            property_type
            + " | "
            + configuration
        )


        property_size = st.number_input(
            "Property size (sq.m)",
            min_value=10.0,
            max_value=7500.0,
            value=80.0,
            step=1.0,
            key="estimate_size"
        )


        contract_type = st.selectbox(
            "Contract type",
            categories[
                "contract_types"
            ],
            key="estimate_contract"
        )


        contract_month = st.selectbox(
            "Contract start month",
            options=list(
                month_names.keys()
            ),
            format_func=lambda x:
                month_names[x],
            key="estimate_month"
        )


        contract_year = st.selectbox(
            "Contract year",
            [
                2025,
                2026
            ],
            key="estimate_year"
        )


    with right:

        st.subheader(
            "Rental quote"
        )


        include_quote = st.checkbox(
            "I have a quoted annual rent",
            key="estimate_quote_check"
        )


        quoted_rent = None


        if include_quote:

            quoted_rent = st.number_input(
                "Quoted annual rent (AED)",
                min_value=0.0,
                value=80000.0,
                step=1000.0,
                key="estimate_quote"
            )


        st.write("")


        estimate_clicked = st.button(
            "Estimate rent",
            type="primary",
            use_container_width=True,
            key="estimate_button"
        )


    # ----------------------------------------------
    # RESULTS
    # ----------------------------------------------

    if estimate_clicked:

        result = predict_rent(

            property_size=
                property_size,

            area_name=
                area_name,

            property_config=
                property_config,

            contract_type=
                contract_type,

            contract_year=
                contract_year,

            contract_month=
                contract_month,

            quoted_rent=
                quoted_rent
        )


        st.divider()

        st.subheader(
            "Your result"
        )


        r1, r2, r3 = st.columns(
            3
        )


        with r1:

            st.metric(
                "Estimated annual rent",
                f"AED {result['prediction']:,.0f}"
            )


        with r2:

            st.metric(
                "Lower range",
                f"AED {result['lower']:,.0f}"
            )


        with r3:

            st.metric(
                "Upper range",
                f"AED {result['upper']:,.0f}"
            )


        if include_quote:

            st.divider()

            position = result[
                "position"
            ]


            if position == \
                "Within predicted range":

                st.success(
                    "Your quoted rent is within the model's predicted range."
                )


            elif position == \
                "Above predicted range":

                st.warning(
                    "Your quoted rent is above the model's predicted range."
                )


            else:

                st.info(
                    "Your quoted rent is below the model's predicted range."
                )


            q1, q2 = st.columns(
                2
            )


            with q1:

                st.metric(
                    "Quoted rent",
                    f"AED {result['quoted_rent']:,.0f}"
                )


            with q2:

                st.metric(
                    "Difference from estimate",
                    f"AED {result['difference']:,.0f}",
                    f"{result['difference_percent']:+.1f}%"
                )


        if contract_year > 2025:

            st.warning(
                """
                The model was formally evaluated on
                2025 contracts. A 2026 estimate
                extends beyond the final test period.
                """
            )


# ==================================================
# BUDGET EXPLORER PAGE
# ==================================================

elif st.session_state.page == "budget":


    back_button()

    st.title(
        "📍 Budget Explorer"
    )

    st.write(
        """
        Enter your annual budget and preferred
        property requirements to explore Dubai
        areas that may fit them.
        """
    )

    st.divider()


    c1, c2 = st.columns(
        2,
        gap="large"
    )


    with c1:

        budget = st.number_input(
            "Maximum annual budget (AED)",
            min_value=5000.0,
            value=100000.0,
            step=5000.0,
            key="budget_amount_v5"
        )


        budget_property_type = (
            st.selectbox(
                "Property type",
                sorted(
                    config_map.keys()
                ),
                key="budget_type_v5"
            )
        )


        budget_configuration = (
            st.selectbox(
                "Bedrooms / configuration",
                config_map[
                    budget_property_type
                ],
                key="budget_config_v5"
            )
        )


        budget_property_config = (
            budget_property_type
            + " | "
            + budget_configuration
        )


        budget_size = st.number_input(
            "Property size (sq.m)",
            min_value=10.0,
            max_value=7500.0,
            value=80.0,
            step=1.0,
            key="budget_size_v5"
        )


    with c2:

        budget_contract_type = (
            st.selectbox(
                "Contract type",
                categories[
                    "contract_types"
                ],
                key="budget_contract_v5"
            )
        )


        budget_month = st.selectbox(
            "Contract start month",
            options=list(
                month_names.keys()
            ),
            format_func=lambda x:
                month_names[x],
            key="budget_month_v5"
        )


        budget_year = st.selectbox(
            "Contract year",
            [
                2025,
                2026
            ],
            key="budget_year_v5"
        )


        st.write("")

        budget_clicked = st.button(
            "Find possible areas",
            type="primary",
            use_container_width=True,
            key="budget_button_v5"
        )


    if budget_clicked:


        candidate_areas = (
            categories[
                "areas"
            ]
        )


        area_inputs = pd.DataFrame({

            "actual_area_clean":
                [float(budget_size)]
                * len(candidate_areas),

            "contract_year":
                [int(budget_year)]
                * len(candidate_areas),

            "area_name_en":
                candidate_areas,

            "property_config":
                [budget_property_config]
                * len(candidate_areas),

            "contract_month":
                [int(budget_month)]
                * len(candidate_areas),

            "contract_reg_type_en":
                [budget_contract_type]
                * len(candidate_areas)
        })


        encoded_areas = (
            preprocessor.transform(
                area_inputs
            )
        )


        area_predictions = (
            model.predict(
                encoded_areas
            )
        )


        half_width = float(
            metadata[
                "prediction_range_half_width_AED"
            ]
        )


        suggestions = pd.DataFrame({

            "Area":
                candidate_areas,

            "Estimated rent":
                area_predictions
        })


        suggestions[
            "Lower range"
        ] = np.maximum(

            0,

            suggestions[
                "Estimated rent"
            ]
            - half_width
        )


        suggestions[
            "Upper range"
        ] = (

            suggestions[
                "Estimated rent"
            ]
            + half_width
        )


        relevant_support = (

            support_data[
                (
                    support_data[
                        "property_config"
                    ]
                    ==
                    budget_property_config
                )
                &
                (
                    support_data[
                        "contract_reg_type_en"
                    ]
                    ==
                    budget_contract_type
                )
            ]

            [
                [
                    "area_name_en",
                    "recent_records"
                ]
            ]

            .rename(
                columns={
                    "area_name_en":
                        "Area"
                }
            )
        )


        suggestions = (
            suggestions.merge(
                relevant_support,
                on="Area",
                how="left"
            )
        )


        suggestions[
            "recent_records"
        ] = (

            suggestions[
                "recent_records"
            ]

            .fillna(0)

            .astype(int)
        )


        suggestions[
            "Budget fit"
        ] = np.select(

            [
                suggestions[
                    "Upper range"
                ] <= budget,

                suggestions[
                    "Estimated rent"
                ] <= budget,

                suggestions[
                    "Lower range"
                ] <= budget
            ],

            [
                "Range within budget",
                "Estimate within budget",
                "Budget overlaps range"
            ],

            default=
                "Above budget"
        )


        possible_areas = (

            suggestions[
                (
                    suggestions[
                        "Budget fit"
                    ]
                    !=
                    "Above budget"
                )
                &
                (
                    suggestions[
                        "recent_records"
                    ]
                    >= 100
                )
            ]

            .copy()
        )


        possible_areas[
            "Difference from budget"
        ] = (

            budget
            -
            possible_areas[
                "Estimated rent"
            ]

        ).abs()


        possible_areas = (

            possible_areas.sort_values(

                [
                    "Difference from budget",
                    "recent_records"
                ],

                ascending=[
                    True,
                    False
                ]
            )
        )


        st.divider()

        st.subheader(
            "Potential areas"
        )


        if len(
            possible_areas
        ) == 0:

            st.warning(
                """
                No areas with at least 100 recent
                comparable contracts met the
                selected criteria.
                """
            )


        else:

            st.write(
                f"""
                **{len(possible_areas)} areas**
                matched the selected criteria.
                """
            )


            display_results = (

                possible_areas[
                    [
                        "Area",
                        "Estimated rent",
                        "Lower range",
                        "Upper range",
                        "Budget fit",
                        "recent_records"
                    ]
                ]

                .head(
                    15
                )

                .copy()
            )


            for column in [
                "Estimated rent",
                "Lower range",
                "Upper range"
            ]:

                display_results[
                    column
                ] = (

                    display_results[
                        column
                    ]

                    .round(0)

                    .astype(int)
                )


            st.dataframe(
                display_results,
                use_container_width=True,
                hide_index=True
            )


            st.caption(
                """
                Results are model-based suggestions,
                not current listings or assessments
                of neighbourhood quality.
                """
            )


        if budget_year > 2025:

            st.warning(
                """
                Suggestions for 2026 extend beyond
                the model's final 2025 test period.
                """
            )




# ==================================================
# STAY VS MOVE PAGE - V6
# ==================================================

elif st.session_state.page == "move":


    back_button()

    st.title(
        "🔄 Stay vs Move"
    )

    st.write(
        """
        Compare your current rental with a possible
        move using the same frozen rent-prediction
        model used elsewhere in the portal.

        The comparison combines modelled rent,
        user-entered rent, moving costs and
        multi-year costs.
        """
    )

    st.divider()


    # ==================================================
    # SECTION 1 - CURRENT HOME
    # ==================================================

    st.header(
        "1. Your current home"
    )

    st.write(
        """
        First enter the details of the property you
        currently live in and the annual rent or
        renewal quote you are considering.
        """
    )


    current_col1, current_col2 = st.columns(
        2,
        gap="large"
    )


    with current_col1:


        current_area = st.selectbox(
            "Current area",
            categories[
                "areas"
            ],
            key="v6_current_area"
        )


        current_property_type = st.selectbox(
            "Current property type",
            sorted(
                config_map.keys()
            ),
            key="v6_current_type"
        )


        current_configuration = st.selectbox(
            "Current bedrooms / configuration",
            config_map[
                current_property_type
            ],
            key="v6_current_config"
        )


        current_property_config = (
            current_property_type
            + " | "
            + current_configuration
        )


        current_size = st.number_input(
            "Current property size (sq.m)",
            min_value=10.0,
            max_value=7500.0,
            value=80.0,
            step=1.0,
            key="v6_current_size"
        )


    with current_col2:


        current_contract_options = (
            categories[
                "contract_types"
            ]
        )


        default_contract_index = 0

        if "Renew" in current_contract_options:

            default_contract_index = (
                current_contract_options
                .index(
                    "Renew"
                )
            )


        current_contract_type = st.selectbox(
            "Current contract type",
            current_contract_options,
            index=default_contract_index,
            key="v6_current_contract"
        )


        current_month = st.selectbox(
            "Current / renewal contract month",
            options=list(
                month_names.keys()
            ),
            format_func=lambda x:
                month_names[x],
            key="v6_current_month"
        )


        current_year = st.selectbox(
            "Current / renewal contract year",
            [
                2025,
                2026
            ],
            key="v6_current_year"
        )


        current_rent = st.number_input(
            "Current annual rent or renewal quote (AED)",
            min_value=0.0,
            value=90000.0,
            step=1000.0,
            key="v6_current_rent"
        )


    # --------------------------------------------------
    # CURRENT PROPERTY MODEL ESTIMATE
    # --------------------------------------------------

    current_model_result = predict_rent(

        property_size=
            current_size,

        area_name=
            current_area,

        property_config=
            current_property_config,

        contract_type=
            current_contract_type,

        contract_year=
            current_year,

        contract_month=
            current_month,

        quoted_rent=
            current_rent
    )


    st.subheader(
        "Current rent check"
    )


    cur1, cur2, cur3 = st.columns(
        3
    )


    with cur1:

        st.metric(
            "Current / renewal rent",
            f"AED {current_rent:,.0f}"
        )


    with cur2:

        st.metric(
            "Model estimate",
            f"AED {current_model_result['prediction']:,.0f}"
        )


    with cur3:

        st.metric(
            "Model range",
            (
                f"AED "
                f"{current_model_result['lower']:,.0f}"
                f" – "
                f"{current_model_result['upper']:,.0f}"
            )
        )


    current_position = (
        current_model_result[
            "position"
        ]
    )


    if current_position == \
        "Within predicted range":

        st.success(
            """
            Your current rent or renewal quote is
            within the modelled prediction range.
            """
        )


    elif current_position == \
        "Above predicted range":

        st.warning(
            """
            Your current rent or renewal quote is
            above the modelled prediction range.
            """
        )


    else:

        st.info(
            """
            Your current rent or renewal quote is
            below the modelled prediction range.
            """
        )


    if current_year > 2025:

        st.warning(
            """
            The model was formally evaluated on
            2025 contracts. A 2026 estimate extends
            beyond the final test period.
            """
        )


    st.divider()


    # ==================================================
    # SECTION 2 - POSSIBLE MOVE
    # ==================================================

    st.header(
        "2. Your possible move"
    )


    move_route = st.radio(

        "How would you like to explore the move?",

        [
            "I know the area I want to move to",
            "Suggest areas that fit my budget"
        ],

        key="v6_move_route"
    )


    move_col1, move_col2 = st.columns(
        2,
        gap="large"
    )


    # --------------------------------------------------
    # COMMON MOVE REQUIREMENTS
    # --------------------------------------------------

    with move_col1:


        move_budget = st.number_input(
            "Maximum annual rental budget (AED)",
            min_value=5000.0,
            value=85000.0,
            step=5000.0,
            key="v6_move_budget"
        )


        move_property_type = st.selectbox(
            "Desired property type",
            sorted(
                config_map.keys()
            ),
            key="v6_move_type"
        )


        move_configuration = st.selectbox(
            "Desired bedrooms / configuration",
            config_map[
                move_property_type
            ],
            key="v6_move_config"
        )


        move_property_config = (
            move_property_type
            + " | "
            + move_configuration
        )


        move_size = st.number_input(
            "Desired property size (sq.m)",
            min_value=10.0,
            max_value=7500.0,
            value=80.0,
            step=1.0,
            key="v6_move_size"
        )


    with move_col2:


        move_contract_options = (
            categories[
                "contract_types"
            ]
        )


        move_default_index = 0

        if "New" in move_contract_options:

            move_default_index = (
                move_contract_options
                .index(
                    "New"
                )
            )


        move_contract_type = st.selectbox(
            "New property contract type",
            move_contract_options,
            index=move_default_index,
            key="v6_move_contract"
        )


        move_month = st.selectbox(
            "Expected move month",
            options=list(
                month_names.keys()
            ),
            format_func=lambda x:
                month_names[x],
            key="v6_move_month"
        )


        move_year = st.selectbox(
            "Expected move year",
            [
                2025,
                2026
            ],
            key="v6_move_year"
        )


    selected_move_area = None

    move_model_result = None

    move_support_records = None


    # ==================================================
    # ROUTE A - USER KNOWS AREA
    # ==================================================

    if move_route == \
        "I know the area I want to move to":


        selected_move_area = st.selectbox(
            "Area you want to move to",
            categories[
                "areas"
            ],
            key="v6_known_move_area"
        )


        move_model_result = predict_rent(

            property_size=
                move_size,

            area_name=
                selected_move_area,

            property_config=
                move_property_config,

            contract_type=
                move_contract_type,

            contract_year=
                move_year,

            contract_month=
                move_month
        )


        support_match = (

            support_data[
                (
                    support_data[
                        "area_name_en"
                    ]
                    ==
                    selected_move_area
                )
                &
                (
                    support_data[
                        "property_config"
                    ]
                    ==
                    move_property_config
                )
                &
                (
                    support_data[
                        "contract_reg_type_en"
                    ]
                    ==
                    move_contract_type
                )
            ]
        )


        if len(
            support_match
        ) > 0:

            move_support_records = int(
                support_match[
                    "recent_records"
                ]
                .iloc[0]
            )

        else:

            move_support_records = 0


        st.subheader(
            "Possible move estimate"
        )


        mv1, mv2, mv3 = st.columns(
            3
        )


        with mv1:

            st.metric(
                "Model estimate",
                f"AED {move_model_result['prediction']:,.0f}"
            )


        with mv2:

            st.metric(
                "Lower range",
                f"AED {move_model_result['lower']:,.0f}"
            )


        with mv3:

            st.metric(
                "Upper range",
                f"AED {move_model_result['upper']:,.0f}"
            )


        if (
            move_model_result[
                "prediction"
            ]
            <=
            move_budget
        ):

            st.success(
                """
                The model estimate is within your
                selected annual budget.
                """
            )

        elif (
            move_model_result[
                "lower"
            ]
            <=
            move_budget
        ):

            st.warning(
                """
                Your budget is below the model's
                central estimate but overlaps the
                prediction range.
                """
            )

        else:

            st.warning(
                """
                This property is above your selected
                budget based on the modelled range.
                """
            )


        st.caption(
            f"""
            Recent supporting contracts for this
            area/property/contract combination:
            {move_support_records:,}
            """
        )


    # ==================================================
    # ROUTE B - SUGGEST AREAS
    # ==================================================

    else:


        st.write(
            """
            The model will compare the selected
            property requirements across supported
            Dubai areas and return areas that overlap
            your budget.
            """
        )


        if st.button(
            "Find areas for my move",
            type="primary",
            use_container_width=True,
            key="v6_find_move_areas"
        ):


            candidate_areas = (
                categories[
                    "areas"
                ]
            )


            move_area_inputs = pd.DataFrame({

                "actual_area_clean":
                    [float(move_size)]
                    * len(candidate_areas),

                "contract_year":
                    [int(move_year)]
                    * len(candidate_areas),

                "area_name_en":
                    candidate_areas,

                "property_config":
                    [move_property_config]
                    * len(candidate_areas),

                "contract_month":
                    [int(move_month)]
                    * len(candidate_areas),

                "contract_reg_type_en":
                    [move_contract_type]
                    * len(candidate_areas)
            })


            move_encoded = (
                preprocessor.transform(
                    move_area_inputs
                )
            )


            move_predictions = (
                model.predict(
                    move_encoded
                )
            )


            half_width = float(
                metadata[
                    "prediction_range_half_width_AED"
                ]
            )


            move_suggestions = pd.DataFrame({

                "Area":
                    candidate_areas,

                "Estimated rent":
                    move_predictions
            })


            move_suggestions[
                "Lower range"
            ] = np.maximum(

                0,

                move_suggestions[
                    "Estimated rent"
                ]
                - half_width
            )


            move_suggestions[
                "Upper range"
            ] = (

                move_suggestions[
                    "Estimated rent"
                ]
                + half_width
            )


            relevant_support = (

                support_data[
                    (
                        support_data[
                            "property_config"
                        ]
                        ==
                        move_property_config
                    )
                    &
                    (
                        support_data[
                            "contract_reg_type_en"
                        ]
                        ==
                        move_contract_type
                    )
                ]

                [
                    [
                        "area_name_en",
                        "recent_records"
                    ]
                ]

                .rename(
                    columns={
                        "area_name_en":
                            "Area"
                    }
                )
            )


            move_suggestions = (
                move_suggestions.merge(

                    relevant_support,

                    on="Area",

                    how="left"
                )
            )


            move_suggestions[
                "recent_records"
            ] = (

                move_suggestions[
                    "recent_records"
                ]

                .fillna(0)

                .astype(int)
            )


            move_suggestions[
                "Budget fit"
            ] = np.select(

                [
                    move_suggestions[
                        "Upper range"
                    ]
                    <=
                    move_budget,

                    move_suggestions[
                        "Estimated rent"
                    ]
                    <=
                    move_budget,

                    move_suggestions[
                        "Lower range"
                    ]
                    <=
                    move_budget
                ],

                [
                    "Range within budget",
                    "Estimate within budget",
                    "Budget overlaps range"
                ],

                default=
                    "Above budget"
            )


            supported_move_areas = (

                move_suggestions[
                    (
                        move_suggestions[
                            "Budget fit"
                        ]
                        !=
                        "Above budget"
                    )
                    &
                    (
                        move_suggestions[
                            "recent_records"
                        ]
                        >= 100
                    )
                ]

                .copy()
            )


            supported_move_areas[
                "Difference from budget"
            ] = (

                move_budget

                -
                supported_move_areas[
                    "Estimated rent"
                ]

            ).abs()


            supported_move_areas = (

                supported_move_areas
                .sort_values(

                    [
                        "Difference from budget",
                        "recent_records"
                    ],

                    ascending=[
                        True,
                        False
                    ]
                )

                .head(
                    10
                )

                .reset_index(
                    drop=True
                )
            )


            st.session_state[
                "v6_move_suggestions"
            ] = supported_move_areas


        # ----------------------------------------------
        # DISPLAY SUGGESTIONS
        # ----------------------------------------------

        if (
            "v6_move_suggestions"
            in
            st.session_state
        ):


            supported_move_areas = (
                st.session_state[
                    "v6_move_suggestions"
                ]
            )


            if len(
                supported_move_areas
            ) == 0:

                st.warning(
                    """
                    No area with at least 100 recent
                    comparable contracts matched the
                    selected budget and property
                    requirements.
                    """
                )


            else:


                st.subheader(
                    "Suggested areas"
                )


                suggestion_display = (
                    supported_move_areas[
                        [
                            "Area",
                            "Estimated rent",
                            "Lower range",
                            "Upper range",
                            "Budget fit",
                            "recent_records"
                        ]
                    ]

                    .copy()
                )


                for column in [
                    "Estimated rent",
                    "Lower range",
                    "Upper range"
                ]:

                    suggestion_display[
                        column
                    ] = (

                        suggestion_display[
                            column
                        ]

                        .round(0)

                        .astype(int)
                    )


                st.dataframe(
                    suggestion_display,
                    use_container_width=True,
                    hide_index=True
                )


                selected_move_area = (
                    st.selectbox(
                        "Choose an area to use in the comparison",
                        supported_move_areas[
                            "Area"
                        ]
                        .tolist(),
                        key="v6_selected_suggested_area"
                    )
                )


                selected_row = (

                    supported_move_areas[
                        supported_move_areas[
                            "Area"
                        ]
                        ==
                        selected_move_area
                    ]

                    .iloc[0]
                )


                move_model_result = predict_rent(

                    property_size=
                        move_size,

                    area_name=
                        selected_move_area,

                    property_config=
                        move_property_config,

                    contract_type=
                        move_contract_type,

                    contract_year=
                        move_year,

                    contract_month=
                        move_month
                )


                move_support_records = int(
                    selected_row[
                        "recent_records"
                    ]
                )


                st.info(
                    f"""
                    **Selected area:** {selected_move_area}

                    Model estimate:
                    **AED {move_model_result['prediction']:,.0f}**

                    Prediction range:
                    **AED {move_model_result['lower']:,.0f}
                    to AED {move_model_result['upper']:,.0f}**

                    Recent supporting contracts:
                    **{move_support_records:,}**
                    """
                )


    if move_year > 2025:

        st.warning(
            """
            The model was formally evaluated on
            2025 contracts. A 2026 move estimate
            extends beyond the final test period.
            """
        )


    # ==================================================
    # MOVE QUOTE OPTION
    # ==================================================

    if move_model_result is not None:


        st.subheader(
            "Optional move quote"
        )


        has_move_quote = st.checkbox(
            "I already have an actual rental quote for the new property",
            key="v6_has_move_quote"
        )


        move_quote = None


        if has_move_quote:

            move_quote = st.number_input(
                "Actual quoted annual rent for the new property (AED)",
                min_value=0.0,
                value=float(
                    round(
                        move_model_result[
                            "prediction"
                        ],
                        -3
                    )
                ),
                step=1000.0,
                key="v6_move_quote"
            )


            if (
                move_quote
                <
                move_model_result[
                    "lower"
                ]
            ):

                st.info(
                    """
                    The move quote is below the
                    modelled prediction range.
                    """
                )


            elif (
                move_quote
                >
                move_model_result[
                    "upper"
                ]
            ):

                st.warning(
                    """
                    The move quote is above the
                    modelled prediction range.
                    """
                )


            else:

                st.success(
                    """
                    The move quote is within the
                    modelled prediction range.
                    """
                )


        # Rent used in financial comparison

        if move_quote is not None:

            comparison_move_rent = (
                float(
                    move_quote
                )
            )

            move_rent_source_text = (
                "actual move quote"
            )

        else:

            comparison_move_rent = float(
                move_model_result[
                    "prediction"
                ]
            )

            move_rent_source_text = (
                "model estimate"
            )


        st.divider()


        # ==================================================
        # SECTION 3 - COST ASSUMPTIONS
        # ==================================================

        st.header(
            "3. Cost assumptions"
        )


        st.write(
            """
            Official fixed charges are entered as
            defaults where available. Other costs can
            be changed to match your actual situation.
            """
        )


        comparison_years = st.selectbox(
            "Comparison period",
            [
                1,
                2,
                3,
                4,
                5
            ],
            format_func=lambda x:
                f"{x} year"
                if x == 1
                else f"{x} years",
            index=2,
            key="v6_comparison_years"
        )


        rate_col1, rate_col2 = st.columns(
            2,
            gap="large"
        )


        with rate_col1:


            stay_growth = st.number_input(
                "Expected annual change in stay rent (%)",
                min_value=-20.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                key="v6_stay_growth"
            )


        with rate_col2:


            move_growth = st.number_input(
                "Expected annual change in move rent (%)",
                min_value=-20.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                key="v6_move_growth"
            )


        # --------------------------------------------------
        # MUNICIPALITY FEE
        # --------------------------------------------------

        include_housing_fee = st.checkbox(
            "Include Dubai municipality housing fee",
            value=True,
            key="v6_housing_fee"
        )


        housing_fee_percent = st.number_input(
            "Municipality housing fee (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="v6_housing_fee_percent",
            disabled=not include_housing_fee
        )


        st.caption(
            """
            Default: 5% of annual rent. This is
            treated as a recurring occupancy cost
            for both Stay and Move.
            """
        )


        st.subheader(
            "One-off moving costs"
        )


        cost1, cost2 = st.columns(
            2,
            gap="large"
        )


        with cost1:


            agency_percent = st.number_input(
                "Agency fee assumption (%)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                key="v6_agency_percent"
            )


            agency_fee = (
                comparison_move_rent
                *
                agency_percent
                /
                100
            )


            st.caption(
                f"""
                Current calculated agency allowance:
                AED {agency_fee:,.0f}.
                The percentage is editable because
                commission is not a fixed government
                charge.
                """
            )


            ejari_fee = st.number_input(
                "Ejari registration (AED)",
                min_value=0.0,
                value=177.75,
                step=1.0,
                key="v6_ejari"
            )


            st.caption(
                """
                Default uses the online/app Ejari
                registration total.
                """
            )


            dewa_activation = st.number_input(
                "DEWA activation (AED)",
                min_value=0.0,
                value=155.0,
                step=5.0,
                key="v6_dewa_activation"
            )


            st.caption(
                """
                Default uses the published residential
                small-meter activation total.
                """
            )


        with cost2:


            moving_company = st.number_input(
                "Moving company / transport (AED)",
                min_value=0.0,
                value=0.0,
                step=500.0,
                key="v6_moving_company"
            )


            st.caption(
                """
                No fixed official rate is assumed
                because this depends on the size and
                type of move.
                """
            )


            setup_admin = st.number_input(
                "Internet / other setup costs (AED)",
                min_value=0.0,
                value=0.0,
                step=250.0,
                key="v6_setup_admin"
            )


            other_move_cost = st.number_input(
                "Other non-refundable moving costs (AED)",
                min_value=0.0,
                value=0.0,
                step=250.0,
                key="v6_other_cost"
            )


        # --------------------------------------------------
        # REFUNDABLE / CASH-FLOW ITEMS
        # --------------------------------------------------

        st.subheader(
            "Refundable or upfront cash requirements"
        )


        st.write(
            """
            These amounts are shown separately and are
            not treated as permanent moving costs.
            """
        )


        move_furnishing = st.selectbox(
            "New property furnishing",
            [
                "Unfurnished",
                "Furnished"
            ],
            key="v6_furnishing"
        )


        suggested_security_percent = (
            5.0
            if move_furnishing
            ==
            "Unfurnished"
            else
            10.0
        )


        security_deposit_percent = (
            st.number_input(
                "Property security deposit assumption (%)",
                min_value=0.0,
                max_value=30.0,
                value=suggested_security_percent,
                step=1.0,
                key="v6_security_percent"
            )
        )


        property_security_deposit = (
            comparison_move_rent
            *
            security_deposit_percent
            /
            100
        )


        # DEWA deposit assumptions

        current_dewa_deposit = (
            4000.0
            if current_property_type
            ==
            "Villa"
            else
            2000.0
        )


        new_dewa_deposit = (
            4000.0
            if move_property_type
            ==
            "Villa"
            else
            2000.0
        )


        suggested_dewa_difference = max(
            0.0,
            new_dewa_deposit
            -
            current_dewa_deposit
        )


        dewa_deposit_difference = (
            st.number_input(
                "Additional DEWA refundable deposit required (AED)",
                min_value=0.0,
                value=float(
                    suggested_dewa_difference
                ),
                step=500.0,
                key="v6_dewa_deposit_difference"
            )
        )


        st.caption(
            f"""
            Planning assumption:
            current property AED
            {current_dewa_deposit:,.0f};
            new property AED
            {new_dewa_deposit:,.0f}.
            Existing DEWA deposits may be transferred,
            so only an additional difference is shown
            by default.
            """
        )


        # ==================================================
        # FINAL COMPARISON BUTTON
        # ==================================================

        st.divider()


        if st.button(
            "Build Stay vs Move report",
            type="primary",
            use_container_width=True,
            key="v6_build_report"
        ):


            # ----------------------------------------------
            # ONE-OFF NON-REFUNDABLE COSTS
            # ----------------------------------------------

            non_refundable_move_costs = (

                agency_fee
                + ejari_fee
                + dewa_activation
                + moving_company
                + setup_admin
                + other_move_cost
            )


            refundable_upfront = (

                property_security_deposit
                + dewa_deposit_difference
            )


            # ----------------------------------------------
            # MULTI-YEAR COSTS
            # ----------------------------------------------

            cumulative_stay = 0.0

            cumulative_move = (
                non_refundable_move_costs
            )


            yearly_rows = []


            break_even_year = None


            for year_number in range(
                1,
                comparison_years + 1
            ):


                stay_year_rent = (

                    current_rent

                    *
                    (
                        1
                        +
                        stay_growth
                        /
                        100
                    )

                    **
                    (
                        year_number
                        -
                        1
                    )
                )


                move_year_rent = (

                    comparison_move_rent

                    *
                    (
                        1
                        +
                        move_growth
                        /
                        100
                    )

                    **
                    (
                        year_number
                        -
                        1
                    )
                )


                if include_housing_fee:

                    stay_housing_fee = (

                        stay_year_rent

                        *
                        housing_fee_percent

                        /
                        100
                    )


                    move_housing_fee = (

                        move_year_rent

                        *
                        housing_fee_percent

                        /
                        100
                    )

                else:

                    stay_housing_fee = 0.0

                    move_housing_fee = 0.0


                stay_year_total = (

                    stay_year_rent
                    +
                    stay_housing_fee
                )


                move_year_total = (

                    move_year_rent
                    +
                    move_housing_fee
                )


                cumulative_stay += (
                    stay_year_total
                )


                cumulative_move += (
                    move_year_total
                )


                if (
                    break_even_year
                    is None
                    and
                    cumulative_move
                    <=
                    cumulative_stay
                ):

                    break_even_year = (
                        year_number
                    )


                yearly_rows.append({

                    "Year":
                        year_number,

                    "Stay rent":
                        stay_year_rent,

                    "Stay housing fee":
                        stay_housing_fee,

                    "Move rent":
                        move_year_rent,

                    "Move housing fee":
                        move_housing_fee,

                    "Cumulative stay":
                        cumulative_stay,

                    "Cumulative move":
                        cumulative_move
                })


            comparison_table = pd.DataFrame(
                yearly_rows
            )


            total_stay = float(
                cumulative_stay
            )


            total_move = float(
                cumulative_move
            )


            difference = (
                total_move
                -
                total_stay
            )


            # ==================================================
            # REPORT
            # ==================================================

            st.divider()

            st.header(
                "4. Stay vs Move report"
            )


            # ----------------------------------------------
            # CURRENT PROPERTY SUMMARY
            # ----------------------------------------------

            st.subheader(
                "Your current property"
            )


            st.write(
                f"""
                **{current_area}**

                {current_property_type},
                {current_configuration},
                {current_size:,.0f} sq.m

                Current / renewal rent:
                **AED {current_rent:,.0f}**

                Model estimate:
                **AED {current_model_result['prediction']:,.0f}**

                Modelled range:
                **AED {current_model_result['lower']:,.0f}
                to AED {current_model_result['upper']:,.0f}**

                Position:
                **{current_model_result['position']}**
                """
            )


            # ----------------------------------------------
            # MOVE SUMMARY
            # ----------------------------------------------

            st.subheader(
                "Possible move"
            )


            st.write(
                f"""
                **{selected_move_area}**

                {move_property_type},
                {move_configuration},
                {move_size:,.0f} sq.m

                Model estimate:
                **AED {move_model_result['prediction']:,.0f}**

                Modelled range:
                **AED {move_model_result['lower']:,.0f}
                to AED {move_model_result['upper']:,.0f}**

                Annual rent used in the cost comparison:
                **AED {comparison_move_rent:,.0f}**
                ({move_rent_source_text})

                Selected budget:
                **AED {move_budget:,.0f}**
                """
            )


            # ----------------------------------------------
            # COST CARDS
            # ----------------------------------------------

            st.subheader(
                f"{comparison_years}-year financial comparison"
            )


            result1, result2, result3 = (
                st.columns(
                    3
                )
            )


            with result1:

                st.metric(
                    "Stay",
                    f"AED {total_stay:,.0f}"
                )


            with result2:

                st.metric(
                    "Move",
                    f"AED {total_move:,.0f}"
                )


            with result3:

                st.metric(
                    "Difference",
                    f"AED {abs(difference):,.0f}"
                )


            if difference > 0:

                st.success(
                    f"""
                    Based on the entered assumptions,
                    **staying is approximately
                    AED {abs(difference):,.0f} lower
                    in financial cost** over
                    {comparison_years}
                    year{"s" if comparison_years > 1 else ""}.
                    """
                )


            elif difference < 0:

                st.success(
                    f"""
                    Based on the entered assumptions,
                    **moving is approximately
                    AED {abs(difference):,.0f} lower
                    in financial cost** over
                    {comparison_years}
                    year{"s" if comparison_years > 1 else ""}.
                    """
                )


            else:

                st.info(
                    """
                    Stay and Move have approximately
                    the same financial cost under the
                    entered assumptions.
                    """
                )


            # ----------------------------------------------
            # BREAK EVEN
            # ----------------------------------------------

            if break_even_year is not None:

                st.info(
                    f"""
                    **Estimated break-even point:**
                    Year {break_even_year}.

                    By this point, the cumulative Move
                    cost is no higher than the
                    cumulative Stay cost under the
                    assumptions entered.
                    """
                )

            else:

                st.info(
                    f"""
                    **No move break-even was reached
                    within the selected
                    {comparison_years}-year period.**
                    """
                )


            # ----------------------------------------------
            # MOVING COST BREAKDOWN
            # ----------------------------------------------

            st.subheader(
                "Moving cost breakdown"
            )


            cost_breakdown = pd.DataFrame({

                "Item": [
                    "Agency fee",
                    "Ejari",
                    "DEWA activation",
                    "Moving company / transport",
                    "Internet / setup",
                    "Other costs"
                ],

                "Amount_AED": [
                    agency_fee,
                    ejari_fee,
                    dewa_activation,
                    moving_company,
                    setup_admin,
                    other_move_cost
                ]
            })


            cost_breakdown[
                "Amount_AED"
            ] = (

                cost_breakdown[
                    "Amount_AED"
                ]

                .round(0)

                .astype(int)
            )


            st.dataframe(
                cost_breakdown,
                use_container_width=True,
                hide_index=True
            )


            st.write(
                f"""
                **Total non-refundable / planning
                moving costs:**
                AED {non_refundable_move_costs:,.0f}
                """
            )


            # ----------------------------------------------
            # REFUNDABLE CASH
            # ----------------------------------------------

            st.subheader(
                "Upfront refundable cash requirements"
            )


            ref1, ref2, ref3 = st.columns(
                3
            )


            with ref1:

                st.metric(
                    "Property deposit",
                    f"AED {property_security_deposit:,.0f}"
                )


            with ref2:

                st.metric(
                    "Additional DEWA deposit",
                    f"AED {dewa_deposit_difference:,.0f}"
                )


            with ref3:

                st.metric(
                    "Total refundable cash",
                    f"AED {refundable_upfront:,.0f}"
                )


            st.caption(
                """
                Refundable deposits are shown
                separately and are not counted as
                permanent moving costs.
                """
            )


            # ----------------------------------------------
            # YEARLY TABLE
            # ----------------------------------------------

            st.subheader(
                "Year-by-year comparison"
            )


            display_yearly = (
                comparison_table.copy()
            )


            for column in [
                "Stay rent",
                "Stay housing fee",
                "Move rent",
                "Move housing fee",
                "Cumulative stay",
                "Cumulative move"
            ]:

                display_yearly[
                    column
                ] = (

                    display_yearly[
                        column
                    ]

                    .round(0)

                    .astype(int)
                )


            st.dataframe(
                display_yearly,
                use_container_width=True,
                hide_index=True
            )


            # ----------------------------------------------
            # LIMITATIONS
            # ----------------------------------------------

            with st.expander(
                "How to interpret this comparison"
            ):


                st.write(
                    """
                    This report is a decision-support
                    comparison, not a recommendation
                    to stay or move.

                    The rent estimates are produced
                    by the frozen research model.
                    Actual available properties and
                    asking rents may differ.

                    Future annual rent changes are
                    assumptions entered by the user,
                    not forecasts made by the model.

                    The calculation considers
                    financial costs only. It does not
                    measure commute, lifestyle,
                    property condition, school access,
                    furnishing quality or other
                    personal preferences.

                    Model uncertainty also differs
                    between property groups and
                    locations.
                    """
                )

