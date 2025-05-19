from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from pydantic.v1 import Extra
import requests
from bs4 import BeautifulSoup


class IcdToolInput(BaseModel):
    query: Optional[str] = Field(
        None,
        description="The ICD code"
    )


class IcdTool(BaseTool):
    name: str = "icd_query"
    description: str = "Retrieve information of diseases based on the ICD code"
    args_schema: Type[BaseModel] = IcdToolInput

    class Config:
        extra = Extra.allow

    def __init__(self, base_url: str = "https://gesund.bund.de/icd-code-suche/"):
        super().__init__()
        self.base_url = base_url

    def _run(self, query) -> object:
        try:
            # Original-Input
            transformed_param = query.replace('.', '-')  # 'b06-0'
            # Erstelle die finale URL
            url = f"{self.base_url}{transformed_param}"  # or use params={...} if needed
            response = requests.get(url)
            # Überprüfe, ob die Abfrage erfolgreich war
            if response.status_code == 200:
                # Das HTML-Dokument parsen
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find das erste div, dessen Attribut data-text-key mit der
                # Abfrage (query) übereinstimmt
                div = soup.find('div', attrs={'data-text-key': query})
                if div:
                    return div.get_text()
                else:
                    return "No matching element found."
        except Exception as e:
            return f"Error retrieving website: {str(e)}"
