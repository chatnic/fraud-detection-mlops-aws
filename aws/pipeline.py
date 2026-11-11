"""SageMaker Pipeline: process -> train -> evaluate -> quality gate -> register.

    python aws/pipeline.py --role-arn arn:aws:iam::<acct>:role/<SageMakerRole> \
        --bucket <bucket> --upsert --start

Status: written against the SageMaker Python SDK v2 and not yet executed in an AWS account.

The registered model lands in a Model Package Group as PendingManualApproval; a human
approves it in SageMaker Studio before aws/deploy.py will deploy it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sagemaker.inputs import TrainingInput
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

SRC = str(Path(__file__).resolve().parents[1] / "src")
FRAMEWORK = "1.2-1"
MODEL_GROUP = "card-fraud-detection"


def build(role: str, bucket: str) -> Pipeline:
    sess = PipelineSession(default_bucket=bucket)
    raw_uri = ParameterString("RawTransactionsS3Uri", default_value=f"s3://{bucket}/fraud/raw/transactions.csv")
    min_pr_auc = ParameterFloat("MinPrAuc", default_value=0.75)
    instance = ParameterString("InstanceType", default_value="ml.m5.xlarge")

    # FrameworkProcessor (unlike SKLearnProcessor) uploads source_dir, so the fraud/ package is importable.
    processor = FrameworkProcessor(estimator_cls=SKLearn, framework_version=FRAMEWORK, role=role,
                                   instance_type=instance, instance_count=1, sagemaker_session=sess,
                                   base_job_name="fraud-processing")
    step_process = ProcessingStep(
        name="BuildFeatures",
        step_args=processor.run(
            code="processing_entry.py", source_dir=SRC,
            inputs=[ProcessingInput(source=raw_uri, destination="/opt/ml/processing/input")],
            outputs=[ProcessingOutput(output_name=n, source=f"/opt/ml/processing/{n}")
                     for n in ("train", "validation", "test")],
        ),
    )

    estimator = SKLearn(entry_point="train_entry.py", source_dir=SRC, framework_version=FRAMEWORK,
                        role=role, instance_type=instance, instance_count=1, sagemaker_session=sess,
                        base_job_name="fraud-train", hyperparameters={"alert-rate": 0.005})
    outputs = step_process.properties.ProcessingOutputConfig.Outputs
    step_train = TrainingStep(
        name="TrainModel",
        step_args=estimator.fit({
            "train": TrainingInput(outputs["train"].S3Output.S3Uri, content_type="text/csv"),
            "validation": TrainingInput(outputs["validation"].S3Output.S3Uri, content_type="text/csv"),
        }),
    )

    evaluation = PropertyFile(name="Evaluation", output_name="evaluation", path="evaluation.json")
    step_eval = ProcessingStep(
        name="EvaluateModel",
        step_args=processor.run(
            code="evaluate_entry.py", source_dir=SRC,
            inputs=[ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                                    destination="/opt/ml/processing/model"),
                    ProcessingInput(source=outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test")],
            outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        ),
        property_files=[evaluation],
    )

    model = SKLearnModel(model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts, role=role,
                         entry_point="inference_entry.py", source_dir=SRC, framework_version=FRAMEWORK,
                         sagemaker_session=sess)
    step_register = ModelStep(
        name="RegisterModel",
        step_args=model.register(content_types=["application/json", "text/csv"], response_types=["application/json"],
                                 inference_instances=["ml.m5.large"], transform_instances=["ml.m5.xlarge"],
                                 model_package_group_name=MODEL_GROUP, approval_status="PendingManualApproval"),
    )

    gate = ConditionStep(
        name="QualityGate",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=step_eval.name, property_file=evaluation, json_path="metrics.pr_auc.value"),
            right=min_pr_auc)],
        if_steps=[step_register],
        else_steps=[FailStep(name="BelowQualityBar", error_message="PR AUC below the minimum; model not registered.")],
    )

    return Pipeline(name="card-fraud-training", parameters=[raw_uri, min_pr_auc, instance],
                    steps=[step_process, step_train, step_eval, gate], sagemaker_session=sess)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role-arn", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--upsert", action="store_true")
    p.add_argument("--start", action="store_true")
    a = p.parse_args()
    pipe = build(a.role_arn, a.bucket)
    if a.upsert:
        pipe.upsert(role_arn=a.role_arn)
    if a.start:
        print("Started:", pipe.start().arn)
    if not (a.upsert or a.start):
        print(pipe.definition())


if __name__ == "__main__":
    main()
