"""Deploy the latest *approved* model package to a real-time endpoint, with data capture
switched on so monitoring can compare live traffic with training data.

    python aws/deploy.py --role-arn <role> --bucket <bucket> --endpoint card-fraud-rt

Status: written against the SageMaker Python SDK v2 and not yet executed in an AWS account.
"""

from __future__ import annotations

import argparse

import boto3
import sagemaker
from sagemaker import ModelPackage
from sagemaker.model_monitor import DataCaptureConfig

MODEL_GROUP = "card-fraud-detection"


def latest_approved(sm_client) -> str:
    pkgs = sm_client.list_model_packages(ModelPackageGroupName=MODEL_GROUP, ModelApprovalStatus="Approved",
                                         SortBy="CreationTime", SortOrder="Descending", MaxResults=1)
    if not pkgs["ModelPackageSummaryList"]:
        raise SystemExit(f"No approved model packages in {MODEL_GROUP}; approve one in SageMaker Studio first.")
    return pkgs["ModelPackageSummaryList"][0]["ModelPackageArn"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role-arn", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--endpoint", default="card-fraud-rt")
    p.add_argument("--instance-type", default="ml.m5.large")
    a = p.parse_args()

    sess = sagemaker.Session(default_bucket=a.bucket)
    arn = latest_approved(boto3.client("sagemaker"))
    model = ModelPackage(role=a.role_arn, model_package_arn=arn, sagemaker_session=sess)
    model.deploy(
        initial_instance_count=1,
        instance_type=a.instance_type,
        endpoint_name=a.endpoint,
        data_capture_config=DataCaptureConfig(enable_capture=True, sampling_percentage=100,
                                              destination_s3_uri=f"s3://{a.bucket}/fraud/data-capture"),
    )
    print(f"Deployed {arn} to endpoint {a.endpoint}")


if __name__ == "__main__":
    main()
