from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5432/roof_measure")
    redis_url: str = Field(default="redis://localhost:6379/0")

    nrw_wcs_dop: str = "https://www.wcs.nrw.de/geobasis/wcs_nw_dop"
    nrw_wcs_ndom: str = "https://www.wcs.nrw.de/geobasis/wcs_nw_ndom"
    nrw_wfs_alkis: str = "https://www.wfs.nrw.de/geobasis/wfs_nw_alkis_vereinfacht"

    nominatim_base: str = "https://nominatim.openstreetmap.org"
    nominatim_user_agent: str = "roof-measure-app/0.1"

    cors_origins: str = "http://localhost:3000"
    log_level: str = "info"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
