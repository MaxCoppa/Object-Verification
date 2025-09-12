import pandas as pd
from datetime import datetime, timedelta
from astral.sun import sun
from astral import LocationInfo


def create_city_location(
    name="Paris",
    region="France",
    timezone="Europe/Paris",
    latitude=48.8566,
    longitude=2.3522,
):

    return LocationInfo(
        name=name,
        region=region,
        timezone=timezone,
        latitude=latitude,
        longitude=longitude,
    )


def is_day_sun(tsp, sunrise, sunset):

    return ((sunrise + timedelta(hours=1)) <= tsp) & (
        tsp <= (sunset - timedelta(hours=1))
    )


def is_night_sun(tsp, sunrise, sunset):

    return (tsp <= (sunrise - timedelta(hours=1))) | (
        (sunset + timedelta(hours=1)) <= tsp
    )


def time_period(tsp, city=create_city_location()):
    """
    Returns whether a timestamp is in the 'day' or 'night' period for a location
    using suntime.
    """
    tsp = tsp.tz_localize(city.timezone)
    sun_times = sun(city.observer, date=tsp)

    sunrise = sun_times["sunrise"]
    sunset = sun_times["sunset"]
    day = is_day_sun(tsp, sunrise, sunset)
    night = is_night_sun(tsp, sunrise, sunset)

    return day, night


def time_difference_within_hour(tsp1, tsp2):
    """
    Returns True if two timestamps are within ±60 minutes, considering wraparound.
    """

    diff = abs((tsp1 - tsp2).total_seconds()) / 60
    diff = min(diff, 1440 - diff)

    return diff <= 60


def check_time_using_suntime(tsp1, tsp2):
    """
    Checks if two timestamps are in the same time period and close enough in time.
    """

    day1, night1 = time_period(tsp1)
    day2, night2 = time_period(tsp2)
    test_hour = time_difference_within_hour(tsp1, tsp2)

    return (day1 & day2) | (night1 & night2)


def is_day_hour(tsp):
    x = tsp.hour
    return 8 <= x <= 18


def is_night_hour(tsp):
    x = tsp.hour
    return 21 <= x or x <= 6


def check_time_using_hours(tsp1, tsp2):
    """
    Simple hour-based rule: same period (day or night) and within 1 hour.
    """
    x_day, y_day = is_day_hour(tsp1), is_day_hour(tsp2)
    x_night, y_night = is_night_hour(tsp1), is_night_hour(tsp2)
    return (x_day and y_day) or (x_night and y_night)
