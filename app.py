
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

ARTIFACT_DIR = Path(".")

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
# STAY VS MOVE PAGE
# ==================================================

elif st.session_state.page == "move":


    back_button()

    st.title(
        "🔄 Stay vs Move"
    )

    st.write(
        """
        Compare the estimated financial cost of
        renewing your current rental with moving to
        another property.
        """
    )

    st.divider()


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
        key="move_years_v5"
    )


    stay_col, move_col = st.columns(
        2,
        gap="large"
    )


    # ----------------------------------------------
    # STAY
    # ----------------------------------------------

    with stay_col:

        st.subheader(
            "Stay"
        )


        renewal_rent = st.number_input(
            "Annual renewal rent (AED)",
            min_value=0.0,
            value=90000.0,
            step=1000.0,
            key="stay_rent_v5"
        )


        stay_rent_change = (
            st.number_input(
                "Expected annual rent change (%)",
                min_value=-20.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                key="stay_change_v5"
            )
        )


    # ----------------------------------------------
    # MOVE
    # ----------------------------------------------

    with move_col:

        st.subheader(
            "Move"
        )


        alternative_rent = st.number_input(
            "Alternative annual rent (AED)",
            min_value=0.0,
            value=85000.0,
            step=1000.0,
            key="alternative_rent_v5"
        )


        move_rent_change = (
            st.number_input(
                "Expected annual rent change (%)",
                min_value=-20.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                key="move_change_v5"
            )
        )


    st.subheader(
        "Moving costs"
    )


    m1, m2, m3 = st.columns(
        3
    )


    with m1:

        agent_fee = st.number_input(
            "Agent fee (AED)",
            min_value=0.0,
            value=0.0,
            step=500.0,
            key="agent_v5"
        )


        moving_cost = st.number_input(
            "Moving cost (AED)",
            min_value=0.0,
            value=0.0,
            step=500.0,
            key="moving_v5"
        )


    with m2:

        setup_cost = st.number_input(
            "Setup / administration (AED)",
            min_value=0.0,
            value=0.0,
            step=500.0,
            key="setup_v5"
        )


        deposit_loss = st.number_input(
            "Non-recoverable deposit cost (AED)",
            min_value=0.0,
            value=0.0,
            step=500.0,
            key="deposit_v5"
        )


    with m3:

        other_cost = st.number_input(
            "Other costs (AED)",
            min_value=0.0,
            value=0.0,
            step=500.0,
            key="other_v5"
        )


    compare_clicked = st.button(
        "Compare Stay vs Move",
        type="primary",
        use_container_width=True,
        key="move_compare_v5"
    )


    if compare_clicked:


        one_off_move_costs = (
            agent_fee
            + moving_cost
            + setup_cost
            + deposit_loss
            + other_cost
        )


        total_stay = 0.0

        total_move = (
            one_off_move_costs
        )


        yearly_rows = []


        for year_number in range(
            1,
            comparison_years + 1
        ):


            stay_year_rent = (

                renewal_rent

                * (
                    1
                    +
                    stay_rent_change
                    / 100
                )

                ** (
                    year_number - 1
                )
            )


            move_year_rent = (

                alternative_rent

                * (
                    1
                    +
                    move_rent_change
                    / 100
                )

                ** (
                    year_number - 1
                )
            )


            total_stay += (
                stay_year_rent
            )


            total_move += (
                move_year_rent
            )


            yearly_rows.append({

                "Year":
                    year_number,

                "Stay rent":
                    stay_year_rent,

                "Move rent":
                    move_year_rent,

                "Cumulative stay":
                    total_stay,

                "Cumulative move":
                    total_move
            })


        difference = (
            total_move
            - total_stay
        )


        st.divider()

        st.subheader(
            "Comparison result"
        )


        r1, r2, r3 = st.columns(
            3
        )


        with r1:

            st.metric(
                "Stay",
                f"AED {total_stay:,.0f}"
            )


        with r2:

            st.metric(
                "Move",
                f"AED {total_move:,.0f}"
            )


        with r3:

            st.metric(
                "Difference",
                f"AED {abs(difference):,.0f}"
            )


        if difference > 0:

            st.success(
                f"""
                Staying is approximately
                **AED {abs(difference):,.0f} cheaper**
                over the selected period.
                """
            )


        elif difference < 0:

            st.success(
                f"""
                Moving is approximately
                **AED {abs(difference):,.0f} cheaper**
                over the selected period.
                """
            )


        else:

            st.info(
                """
                The estimated financial cost is
                approximately the same.
                """
            )


        yearly_table = pd.DataFrame(
            yearly_rows
        )


        st.dataframe(
            yearly_table,
            use_container_width=True,
            hide_index=True
        )


        st.caption(
            """
            This comparison considers financial
            inputs only. It does not include factors
            such as commute, lifestyle, property
            condition or inconvenience.
            """
        )
