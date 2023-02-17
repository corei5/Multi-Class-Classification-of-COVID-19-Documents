#URL Lib to query API
from urllib.request import urlopen

#ElementTree to parse XML response
import xml.etree.ElementTree as ET

#Import Tracker
from tqdm.notebook import tqdm


def ncbi_api(pmcids):
    # Base string is the API end point
    api_base_string = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids="
    # List of PMCIDS to send to API end point
    ids_string = ",".join(list(pmcids))
    # Call API with query
    api_query = api_base_string + ids_string

    # Get API response which is a list of dictionaries
    with urlopen(api_query) as response:
        response_content = response.read()
    root = ET.fromstring(response_content)

    # Return
    return [child.attrib for child in root[1:]]