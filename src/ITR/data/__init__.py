"""This module contains classes that create connections to data providers and initializes our system of units"""

import re

import pint
from openscm_units import unit_registry
from pint import set_application_registry

currency_dict = {
    # 'US$':'USD', NOTE: don't try to re-define USD...it leads to infinite recursion
    "€": "EUR",
    "¥": "JPY",  # 円 / 圓
    "£": "GBP",
    "元": "CNY",  # ¥
    "A$": "AUD",
    "C$": "CAD",
    "HK$": "HKD",
    "S$": "SGD",
    "SKr": "SEK",
    "₩": "KRW",  # 원/ 圓
    "NKr": "NOK",
    "NZ$": "NZD",
    "₹": "INR",
    "NT$": "TWD",
    "R": "ZAR",
    "R$": "BRL",
    "DKr": "DKK",
    "zł": "PLN",
    "฿": "THB",
    "₪": "ILS",
    "Rp": "IDR",
    "Kč": "CZK",
    "د.إ": "AED",
    "₺": "TRY",
    "₴": "HRV",
    "₦": "NGN",
    "د.م.": "MAD",
    "RM": "MYR",
}


def escape_currency_symbols(text):
    if "$" in text:
        escaped_text = re.escape(text)
    else:
        escaped_text = text
    if escaped_text[0].isalpha():
        if escaped_text[-1].isalpha():
            return r"\b" + escaped_text + r"\b"
        return r"\b" + escaped_text
    return escaped_text


currency_keep_regexp = re.compile(
    rf"({'|'.join([cur_abbrev for cur_abbrev in currency_dict.values()])})"
)
currency_split_regexp = re.compile(
    rf"(\$|\bUS\$|{'|'.join([escape_currency_symbols(currency_symbol) for currency_symbol in currency_dict])})"
)


def translate_currency_symbols_1(text):
    split_text = re.split(currency_split_regexp, text)
    if len(split_text) <= 1:
        return text
    if len(split_text) & 1 and split_text[-1] != "":
        split_text.append("")
    pairs = zip(split_text[::2], split_text[1::2])
    retval = "".join(
        map(
            lambda x: f"{x[0]}{'USD' if x[1] == '$' or x[1] == 'US$' else currency_dict.get(x[1], '')}",
            pairs,
        )
    )
    # print(f"{text} -> {split_text} -> {retval}")
    return retval


def translate_currency_symbols(text):
    keep_text = re.split(currency_keep_regexp, text)
    if len(keep_text) <= 1:
        return translate_currency_symbols_1(text)
    if len(keep_text) & 1 and keep_text[-1] != "":
        keep_text.append("")
    pairs = zip(keep_text[::2], keep_text[1::2])
    retval = "".join(
        [
            inner
            for outer in pairs
            for inner in [translate_currency_symbols_1(outer[0]), outer[1]]
        ]
    )
    # print(f"{text} -> {keep_text} -> {retval}")
    return retval


# openscm_units doesn't make it easy to set preprocessors.  This is one way to do it.
unit_registry.preprocessors = [
    lambda s1: re.sub(r"passenger.km", "pkm", s1),
    lambda s2: translate_currency_symbols(s2),
]

ureg = unit_registry
set_application_registry(ureg)

# Overwrite what pint/pint/__init__.py initalizes
# # Default Quantity, Unit and Measurement are the ones
# # build in the default registry.
# Quantity = UnitRegistry.Quantity
# Unit = UnitRegistry.Unit
# Measurement = UnitRegistry.Measurement
# Context = UnitRegistry.Context

pint.Quantity = ureg.Quantity
pint.Unit = ureg.Unit
pint.Measurement = ureg.Measurement
pint.Context = ureg.Context

# FIXME: delay loading of pint_pandas until after we've initialized ourselves
from pint_pandas import PintArray, PintType  # noqa E402

Q_ = ureg.Quantity
M_ = ureg.Measurement
PA_ = PintArray
