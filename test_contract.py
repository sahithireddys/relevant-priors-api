from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_contract():
    payload = {
        "challenge_id": "relevant-priors-v1",
        "schema_version": 1,
        "generated_at": "2026-04-16T12:00:00.000Z",
        "cases": [
            {
                "case_id": "1001016",
                "patient_id": "606707",
                "patient_name": "Andrews, Micheal",
                "current_study": {
                    "study_id": "3100042",
                    "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                    "study_date": "2026-03-08"
                },
                "prior_studies": [
                    {
                        "study_id": "2453245",
                        "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                        "study_date": "2020-03-08"
                    },
                    {
                        "study_id": "992654",
                        "study_description": "CT HEAD WITHOUT CNTRST",
                        "study_date": "2021-03-08"
                    }
                ]
            }
        ]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert {p["study_id"] for p in data["predictions"]} == {"2453245", "992654"}
    assert all(isinstance(p["predicted_is_relevant"], bool) for p in data["predictions"])
