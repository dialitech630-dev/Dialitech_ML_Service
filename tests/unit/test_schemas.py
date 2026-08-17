import pytest
from pydantic import ValidationError

from app.schemas.analyze_request import AnalyzeRequest
from app.schemas.anomaly import AnomalyDetection
from app.schemas.pattern import PatternDetection, PatternInfo
from app.schemas.reading import HealthReading
from app.schemas.risk import RiskPrediction
from app.schemas.trend import TrendAnalysis, TrendInfo

pytestmark = pytest.mark.unit


class TestHealthReadingSchema:
    def test_valid_reading(self) -> None:
        reading = HealthReading(
            heartRate=75.0,
            oxygen=97.0,
            activity=40.0,
            timestamp="2026-08-06T10:00:00Z",
        )
        assert reading.heartRate == 75.0
        assert reading.oxygen == 97.0
        assert reading.activity == 40.0

    def test_missing_field(self) -> None:
        with pytest.raises(ValidationError):
            HealthReading(heartRate=75.0, oxygen=97.0, timestamp="2026-08-06T10:00:00Z")

    def test_wrong_type(self) -> None:
        with pytest.raises(ValidationError):
            HealthReading(
                heartRate="not_a_number",
                oxygen=97.0,
                activity=40.0,
                timestamp="2026-08-06T10:00:00Z",
            )


class TestAnalyzeRequestSchema:
    def test_valid_request(self) -> None:
        request = AnalyzeRequest(
            patientId="patient-123",
            windowSize=12,
            readings=[
                HealthReading(
                    heartRate=75.0,
                    oxygen=97.0,
                    activity=40.0,
                    timestamp="2026-08-06T10:00:00Z",
                )
            ],
        )
        assert request.patientId == "patient-123"
        assert request.windowSize == 12
        assert len(request.readings) == 1

    def test_default_window_size(self) -> None:
        request = AnalyzeRequest(
            patientId="patient-123",
            readings=[
                HealthReading(
                    heartRate=75.0,
                    oxygen=97.0,
                    activity=40.0,
                    timestamp="2026-08-06T10:00:00Z",
                )
            ],
        )
        assert request.windowSize == 12

    def test_window_size_minimum(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(
                patientId="patient-123",
                windowSize=0,
                readings=[
                    HealthReading(
                        heartRate=75.0,
                        oxygen=97.0,
                        activity=40.0,
                        timestamp="2026-08-06T10:00:00Z",
                    )
                ],
            )

    def test_window_size_maximum(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(
                patientId="patient-123",
                windowSize=1001,
                readings=[
                    HealthReading(
                        heartRate=75.0,
                        oxygen=97.0,
                        activity=40.0,
                        timestamp="2026-08-06T10:00:00Z",
                    )
                ],
            )

    def test_missing_patient_id(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(
                windowSize=12,
                readings=[
                    HealthReading(
                        heartRate=75.0,
                        oxygen=97.0,
                        activity=40.0,
                        timestamp="2026-08-06T10:00:00Z",
                    )
                ],
            )

    def test_empty_patient_id(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(
                patientId="",
                windowSize=12,
                readings=[
                    HealthReading(
                        heartRate=75.0,
                        oxygen=97.0,
                        activity=40.0,
                        timestamp="2026-08-06T10:00:00Z",
                    )
                ],
            )

    def test_missing_readings(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(patientId="patient-123", windowSize=12)

    def test_empty_readings_list(self) -> None:
        with pytest.raises(ValidationError):
            AnalyzeRequest(
                patientId="patient-123",
                windowSize=12,
                readings=[],
            )

    def test_readings_max_length(self) -> None:
        with pytest.raises(ValidationError):
            readings = [
                HealthReading(
                    heartRate=75.0,
                    oxygen=97.0,
                    activity=40.0,
                    timestamp="2026-08-06T10:00:00Z",
                )
                for _ in range(10001)
            ]
            AnalyzeRequest(
                patientId="patient-123",
                windowSize=12,
                readings=readings,
            )


class TestRiskPredictionSchema:
    def test_valid_risk_prediction(self) -> None:
        risk = RiskPrediction(
            riskScore=0.85,
            riskLevel="HIGH",
            recommendation="Immediate review needed",
        )
        assert risk.riskScore == 0.85
        assert risk.riskLevel == "HIGH"

    def test_all_risk_levels(self) -> None:
        for level in ("LOW", "MEDIUM", "HIGH"):
            risk = RiskPrediction(
                riskScore=0.5,
                riskLevel=level,
                recommendation="test",
            )
            assert risk.riskLevel == level


class TestTrendInfoSchema:
    def test_valid_trend_info(self) -> None:
        trend = TrendInfo(
            direction="ASCENDING",
            slope=1.5,
            confidence=0.8,
        )
        assert trend.direction == "ASCENDING"
        assert trend.slope == 1.5

    def test_all_directions(self) -> None:
        for direction in ("ASCENDING", "DESCENDING", "STABLE"):
            trend = TrendInfo(
                direction=direction,
                slope=0.0,
                confidence=0.5,
            )
            assert trend.direction == direction


class TestTrendAnalysisSchema:
    def test_valid_trend_analysis(self) -> None:
        trend_info = TrendInfo(direction="STABLE", slope=0.0, confidence=0.5)
        analysis = TrendAnalysis(
            heartRate=trend_info,
            oxygen=trend_info,
            activity=trend_info,
        )
        assert analysis.heartRate.direction == "STABLE"
        assert analysis.oxygen.direction == "STABLE"
        assert analysis.activity.direction == "STABLE"


class TestPatternDetectionSchema:
    def test_valid_pattern_detection(self) -> None:
        detection = PatternDetection(
            patternsFound=True,
            patterns=[
                PatternInfo(
                    type="PERIODIC_OSCILLATION",
                    description="Test pattern",
                    confidence=0.75,
                )
            ],
        )
        assert detection.patternsFound is True
        assert len(detection.patterns) == 1

    def test_no_patterns(self) -> None:
        detection = PatternDetection(
            patternsFound=False,
            patterns=[],
        )
        assert detection.patternsFound is False
        assert len(detection.patterns) == 0


class TestAnomalyDetectionSchema:
    def test_valid_anomaly_detection(self) -> None:
        detection = AnomalyDetection(
            anomalyDetected=True,
            anomalyScore=0.85,
            affectedReadingsIndexes=[2, 5],
        )
        assert detection.anomalyDetected is True
        assert detection.anomalyScore == 0.85
        assert detection.affectedReadingsIndexes == [2, 5]

    def test_no_anomaly(self) -> None:
        detection = AnomalyDetection(
            anomalyDetected=False,
            anomalyScore=0.0,
            affectedReadingsIndexes=[],
        )
        assert detection.anomalyDetected is False
        assert len(detection.affectedReadingsIndexes) == 0
