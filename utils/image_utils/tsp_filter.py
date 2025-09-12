import pandas as pd

from .tsp_utils import check_time_using_hours, check_time_using_suntime


def filter_by_valid_hours(df, use_sun_times=False):
    """
    Filters the DataFrame rows where both 'img_tsp' and 'couple_tsp'
    fall within either daytime (8 AM to 6 PM) or nighttime (9 PM to 6 AM).

    Optionally uses sunrise/sunset info based on location if `use_sun_times` is True.
    """

    # Convert timestamps to datetime
    df["img_tsp"] = pd.to_datetime(df["img_tsp"], format="mixed")
    df["couple_tsp"] = pd.to_datetime(df["couple_tsp"], format="mixed")

    if use_sun_times:
        df_filtered = df[
            df.apply(
                lambda row: check_time_using_suntime(row["img_tsp"], row["couple_tsp"]),
                axis=1,
            )
        ].reset_index(drop=True)
    else:
        df_filtered = df[
            df.apply(
                lambda row: check_time_using_hours(row["img_tsp"], row["couple_tsp"]),
                axis=1,
            )
        ].reset_index(drop=True)

    return df_filtered
