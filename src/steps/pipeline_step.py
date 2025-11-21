from abc import ABC, abstractmethod
from typing import Dict, Any

class PipelineStep(ABC):
    """
    Basis-Klasse für alle Pipeline-Schritte.
    Jeder Schritt muss die run()-Methode implementieren.
    """

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt den Schritt aus und gibt ein Dictionary zurück.
        Das Dictionary kann z.B. einen 'status' Key enthalten:
        - 'progress' -> Schritt erfolgreich
        - 'error' -> Schritt fehlgeschlagen
        """
        pass
