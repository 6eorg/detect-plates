import re


def clean_swiss_license_plate(prepared):
    # List of valid Swiss cantonal codes
    canton_codes = [
        "AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR",
        "JU", "LU", "NE", "NW", "OW", "SG", "SH", "SO", "SZ", "TG",
        "TI", "UR", "VD", "VS", "ZG", "ZH"
    ]

    # Normalize the string (convert to uppercase and strip surrounding whitespace)
    prepared = prepared.upper().strip()

    # Check if any of the cantonal codes is in the prepared string
    for code in canton_codes:
        if code in prepared:
            # Extract the part starting from the cantonal code
            start_index = prepared.index(code)
            prepared = prepared[start_index:]
            break
    else:
        return None  # Return None if no cantonal code is found

    # remove last character / number, as it is often ust the flag
    prepared = prepared[:-1]


    # Remove any non-alphanumeric characters except hyphen
    prepared = re.sub(r'[^A-Z0-9-]', '', prepared)

    # Extract the valid license plate format
    match = re.match(r'([A-Z]{2})-?(\d{1,6})', prepared)


    if match:
        canton_code = match.group(1)
        number = match.group(2)

        # Limit the number to a maximum of 6 digits
        if len(number) > 6:
            number = number[:6]

        # Construct the cleaned license plate
        cleaned_plate = f"{canton_code}-{number}"
        return cleaned_plate
    else:
        return None  # Return None if the pattern doesn't match


# Example usage with provided data
plates = [
    "UBE-£6022 8;",  # Example with some common errors
    "SBE-46022 3°",
    "12 SS0dd- 380",
    "re",  # Invalid plate without any canton code
    "VBE-£4072 §",
    "VBE-£2022 8,",
    "TBE-828480¢"
]
