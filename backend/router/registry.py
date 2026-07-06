from backend.router.advisors import (
    GeneralAdvisor, CropAdvisor, DiseaseAdvisor, AnimalAdvisor,
    DairyAdvisor, WeatherAdvisor, MarketAdvisor, VisionAdvisor, PredictionAdvisor
)

class AdvisorRegistry:
    def __init__(self):
        self._registry = {}
        # Auto-register defaults
        self.register(GeneralAdvisor())
        self.register(CropAdvisor())
        self.register(DiseaseAdvisor())
        self.register(AnimalAdvisor())
        self.register(DairyAdvisor())
        self.register(WeatherAdvisor())
        self.register(MarketAdvisor())
        self.register(VisionAdvisor())
        self.register(PredictionAdvisor())

    def register(self, advisor):
        self._registry[advisor.name] = advisor

    def get_advisors(self):
        return list(self._registry.values())

    def get_by_name(self, name: str):
        return self._registry.get(name)
