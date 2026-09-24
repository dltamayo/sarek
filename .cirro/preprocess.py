#!/usr/bin/env python3

from cirro.helpers.preprocess_dataset import PreprocessDataset
from cirro.models.s3_path import S3Path
import boto3
import pandas as pd
import urllib.request
import urllib.error
import json


# Spellings of sex seen in user samplesheets, mapped to the XX/XY encoding sarek
# requires. Anything else (including a blank cell) becomes the given default.
_SEX_ALIASES = {
    'f': 'XX',
    'female': 'XX',
    'xx': 'XX',
    'm': 'XY',
    'male': 'XY',
    'xy': 'XY',
    'na': 'NA',
    'unknown': 'NA',
}


def normalize_sex(value, default: str = 'NA') -> str:
    """Map a samplesheet sex value to sarek's XX/XY/NA encoding."""
    if pd.isna(value):
        return default
    return _SEX_ALIASES.get(str(value).strip().lower(), default)


def make_manifest(ds: PreprocessDataset) -> pd.DataFrame:

    # Filter out any index files that may have been uploaded
    ds.files = ds.files.loc[
        ds.files.apply(
            lambda r: r.get('readType', 'R') == 'R',
            axis=1
        )
    ]

    # Make a wide manifest
    manifest: pd.DataFrame = ds.wide_samplesheet(
        index=["sampleIndex", "sample", "lane", "dataset"],
        columns="read",
        values="file",
        column_prefix="fastq_"
    )
    assert manifest.shape[0] > 0, "No files detected -- error with data ingest"

    # Get the sample metadata (if any)
    # Populate the 'patient' column with the provided value,
    # falling back to the sample ID if missing.
    # Normalize 'sex' to the XX/XY/NA encoding sarek requires, defaulting to XX
    # since alignment-only does not need sex differentiation. Default 'status' to 0.
    samplesheet = ds.samplesheet.reindex(columns=["sample", "patient", "sex", "status"])
    missing_status = samplesheet["status"].isna().sum()
    if missing_status > 0:
        ds.logger.warning(
            f"status not provided for {missing_status} sample(s), defaulting to 0 (normal). "
            "Set status explicitly in the samplesheet if running somatic variant calling downstream."
        )
    samples = (
        samplesheet
        .assign(patient=lambda d: d['patient'].fillna(d['sample']))
        .assign(sex=lambda d: d['sex'].apply(normalize_sex, default='XX'))
        .assign(status=lambda d: d['status'].fillna(0).astype(int))
        .set_index("sample")
    )

    # 1. Use the 'patient' column if provided, falling back to the 'sample'
    # 2. Order the columns
    # 3. Overwrite the 'lane' column to provide a unique value per-row
    # (This is necessary to account for datasets which merge flowcells)
    manifest = (
        manifest
        .set_index("sample")
        .assign(patient=samples["patient"], sex=samples["sex"], status=samples["status"])
        .reset_index()
        .reindex(columns=['patient', 'sample', 'sex', 'status', 'lane', 'fastq_1', 'fastq_2'])
        .assign(lane=[str(i) for i in range(manifest.shape[0])])
    )

    return manifest


def warn_custom_genome_limitations(ds: PreprocessDataset):
    """Warn when a custom BWA genome is used that base recalibration cannot run.

    Custom genome datasets provide only the FASTA + BWA index — no GATK known-sites
    (dbsnp/known_indels) — so base recalibration will fail. Must be called before
    resolve_reference_genome removes ``genome_source``.
    """
    if ds.params.get("genome_source") != "dataset":
        return
    ds.logger.warning(
        "Custom genome selected: GATK known-sites (dbsnp/known_indels) are not available, "
        "so base recalibration cannot run (it is skipped automatically — see "
        "skip_baserecalibration_without_known_sites)."
    )


def skip_baserecalibration_without_known_sites(ds: PreprocessDataset, is_custom_genome: bool):
    """Skip base recalibration when no known-sites resources are available.

    GATK BaseRecalibrator requires at least one of dbsnp/known_indels. iGenomes
    references supply these via the genome config at runtime, so only custom genomes
    need this guard: when neither resource is present in the params, add
    `baserecalibrator` to `skip_tools` so the run does not fail. Call after all params
    are populated (post extra-JSON and schema filter) so user-supplied resources are
    taken into account.
    """
    if not is_custom_genome:
        return
    if ds.params.get("dbsnp") or ds.params.get("known_indels"):
        return
    existing = ds.params.get("skip_tools")
    skip = [t for t in str(existing).split(",") if t] if existing else []
    if "baserecalibrator" not in skip:
        skip.append("baserecalibrator")
    ds.add_param("skip_tools", ",".join(skip), overwrite=True)
    ds.logger.info(
        "No dbsnp/known_indels provided — adding 'baserecalibrator' to skip_tools "
        f"(skip_tools={','.join(skip)})."
    )


def resolve_reference_genome(ds: PreprocessDataset):
    """Wire up the reference based on the iGenomes vs Custom Genome selection.

    For iGenomes the curated ``genome`` key is passed through unchanged, and
    ``aligner`` (bwa-mem/bwa-mem2/parabricks) is a real sarek param that is
    defaulted and left in place for both branches -- unlike ``genome_source``/
    ``bwa_index``/``bwamem2_index``, which are Cirro-form-only fields that must
    never reach Nextflow. For a custom genome the user selects a pre-built BWA
    or BWA-MEM2 index dataset (mutually exclusive in the form, keyed off
    ``aligner``); we point ``--fasta``/``--bwa``-or-``--bwamem2`` at that
    dataset and drop
    ``--genome``/``--igenomes_base``. Both index pipelines publish
    ``genome.fasta`` and their respective flat index files directly into the
    dataset's data directory, so the directory itself serves as the index
    argument (nf-core's bwa/mem and bwamem2/mem modules derive the index
    prefix from the index files present).

    Dropping ``--genome`` is not sufficient on its own: sarek's nextflow.config
    defaults ``genome`` to 'GATK.GRCh38', so every reference param Cirro leaves unset
    (dict, dbsnp, known_indels, intervals, germline_resource, pon, snpeff_db, vep_*)
    would still resolve to GRCh38 iGenomes values and clash with the custom FASTA.
    ``--igenomes_ignore`` empties ``params.genomes``, so every getGenomeAttribute
    lookup returns null and the missing references are derived from the custom FASTA
    instead.
    """
    genome_source = ds.params.get("genome_source")
    ds.remove_param("genome_source", force=True)

    # A real sarek param (unlike genome_source/bwa_index/bwamem2_index below), so it
    # is defaulted and written back once here instead of being removed and re-added
    # separately in each branch.
    aligner = ds.params.get("aligner") or "bwa-mem"
    ds.add_param("aligner", aligner, overwrite=True)

    bwa_index = ds.params.get("bwa_index")
    ds.remove_param("bwa_index", force=True)

    bwamem2_index = ds.params.get("bwamem2_index")
    ds.remove_param("bwamem2_index", force=True)

    if genome_source == "dataset":
        use_bwamem2 = aligner == "bwa-mem2"
        custom_index = bwamem2_index if use_bwamem2 else bwa_index
        if not custom_index:
            raise ValueError(
                f"Custom Genome selected with aligner={aligner!r} but no matching "
                "genome index dataset was provided."
            )
        ds.logger.info(f"genome_source=dataset: using custom {aligner} index at {custom_index}")
        ds.add_param("fasta", f"{custom_index}/genome.fasta", overwrite=True)
        ds.add_param("fasta_fai", f"{custom_index}/genome.fasta.fai", overwrite=True)
        if use_bwamem2:
            ds.add_param("bwamem2", custom_index, overwrite=True)
        else:
            ds.add_param("bwa", custom_index, overwrite=True)
        ds.add_param("igenomes_ignore", True, overwrite=True)
        ds.remove_param("genome", force=True)
        ds.remove_param("igenomes_base", force=True)
    else:
        ds.logger.info(
            f"genome_source=igenomes: genome={ds.params.get('genome')!r}, aligner={aligner!r}"
        )


_VCF_PARAM_PAIRS = (
    ("dbsnp", "dbsnp_tbi"),
    ("known_indels", "known_indels_tbi"),
)


def stage_colliding_vcf_params(ds: PreprocessDataset):
    paths = {
        vcf_param: ds.params[vcf_param]
        for vcf_param, _ in _VCF_PARAM_PAIRS
        if ds.params.get(vcf_param)
    }

    by_name = {}
    for vcf_param, path in paths.items():
        by_name.setdefault(path.rsplit("/", 1)[-1], []).append(vcf_param)

    colliding = {
        vcf_param
        for vcf_params in by_name.values() if len(vcf_params) > 1
        for vcf_param in vcf_params
    }
    if not colliding:
        ds.logger.info("VCF inputs: no file name collisions to resolve")
        return

    ds.logger.info(f"VCF inputs: resolving file name collision between {sorted(colliding)}")

    s3 = boto3.client("s3")
    config_dir = ds.params["input"].rsplit("/", 1)[0]
    for vcf_param, tbi_param in _VCF_PARAM_PAIRS:
        if vcf_param not in colliding:
            continue

        staged_vcf = f"{vcf_param}_{paths[vcf_param].rsplit('/', 1)[-1]}"
        to_stage = [(vcf_param, paths[vcf_param], staged_vcf)]
        if ds.params.get(tbi_param):
            to_stage.append((tbi_param, ds.params[tbi_param], f"{staged_vcf}.tbi"))

        for param, uri, staged_name in to_stage:
            source = S3Path(uri)
            assert source.valid, f"Cannot stage a copy of --{param}: {uri} is not an S3 path"
            staged_uri = f"{config_dir}/{staged_name}"
            dest = S3Path(staged_uri)
            ds.logger.info(f"VCF inputs: staging {uri} as {staged_uri}")
            s3.copy(
                {"Bucket": source.bucket, "Key": source.key},
                dest.bucket,
                dest.key
            )
            ds.add_param(param, staged_uri, overwrite=True)


def require_analysis_type_binding(ds: PreprocessDataset):
    """Fail when the launch payload did not bind to the form's analysis_type block.

    process-form.json declares ``wes`` required within ``analysis_type`` and gives it a
    default, so any launch whose paramJson matches the form supplies it. Its absence means
    the whole block resolved to nothing — and ``intervals`` lives in that same block, so it
    was dropped too, which would silently turn a targeted run into a genome-wide one.

    Nothing else catches this: the form declares no top-level ``required`` and draft-07
    permits additional properties, so a flat parameter dict validates cleanly and then
    binds none of the nested JSONPaths in process-input.json.
    """
    if "wes" in ds.params:
        return

    raise ValueError(
        "Launch parameters did not match the process form: 'wes' is absent, so the "
        "analysis_type block bound nothing and 'intervals' was dropped with it. "
        "paramJson must nest parameters exactly as process-form.json declares them "
        "(analysis_type.wes, analysis_type.intervals, "
        "analysis_type.genome_selection.igenomes.genome, advanced_options.*, "
        "read_trimming_options.*); a flat parameter dict binds nothing."
    )


_SCHEMA_REPO = "dltamayo/sarek"
_SCHEMA_REF = "cirro-config/parabricks-known-sites-fix"

# Params set by Cirro infrastructure or computed by this script that must not
# be overridden by user-supplied extra JSON.
_PROTECTED_PARAMS = frozenset({
    "input",
    "outdir",
    "igenomes_base",
    "igenomes_ignore",  # set by resolve_reference_genome for custom genomes
    "vep_cache",
    "snpeff_cache",
    "monochrome_logs",
    "compute_multiplier",  # computed from wes
    "wes",              # consumed to compute compute_multiplier before extra JSON is applied
    "intervals",        # consumed to set no_intervals before extra JSON is applied
})

# compute_multiplier and optical_duplicate_pixel_distance are consumed by process-compute.config
# (not the nf-core schema). igenomes_base, vep_cache, snpeff_cache, and monochrome_logs are
# Cirro-pinned S3 paths that must survive schema filtering regardless of sarek version.
_CIRRO_PASSTHROUGH_PARAMS = frozenset({
    "compute_multiplier",
    "optical_duplicate_pixel_distance",
    "igenomes_base",
    "vep_cache",
    "snpeff_cache",
    "monochrome_logs",
})

_MAX_EXTRA_JSON_BYTES = 10_000


def apply_extra_json_params(ds: PreprocessDataset):
    """Parse the extra_params_json textarea and merge user-supplied params into the workflow.

    Applied before filter_params_by_schema so that unrecognized keys are automatically
    removed in the subsequent schema validation step.
    """
    extra_json_str = (ds.params.get("extra_params_json") or "").strip()
    ds.remove_param("extra_params_json", force=True)

    if not extra_json_str:
        ds.logger.info("extra_params_json: no extra parameters provided")
        return

    if len(extra_json_str) > _MAX_EXTRA_JSON_BYTES:
        ds.logger.warning(
            f"extra_params_json: payload too large ({len(extra_json_str):,} chars), skipping"
        )
        return

    ds.logger.info("extra_params_json: parsing user-supplied JSON parameters")

    try:
        extra_params = json.loads(extra_json_str)
    except json.JSONDecodeError as e:
        ds.logger.warning(f"extra_params_json: JSON parse error ({e}), skipping")
        return

    if not isinstance(extra_params, dict):
        ds.logger.warning(
            f"extra_params_json: expected a JSON object but got {type(extra_params).__name__}, skipping"
        )
        return

    ds.logger.info(f"extra_params_json: found {len(extra_params)} parameter(s)")

    applied, skipped = 0, 0
    for key, value in extra_params.items():
        if not isinstance(key, str) or not key.strip():
            ds.logger.warning(f"extra_params_json: skipping invalid key {key!r}")
            skipped += 1
            continue
        if key in _PROTECTED_PARAMS:
            ds.logger.warning(f"extra_params_json: skipping protected parameter '{key}'")
            skipped += 1
            continue
        ds.logger.info(f"extra_params_json: applying {key}={value!r}")
        ds.add_param(key, value, overwrite=True)
        applied += 1

    ds.logger.info(
        f"extra_params_json: applied {applied} parameter(s), skipped {skipped} protected/invalid"
    )


def filter_params_by_schema(ds: PreprocessDataset):
    url = f"https://raw.githubusercontent.com/{_SCHEMA_REPO}/{_SCHEMA_REF}/nextflow_schema.json"
    ds.logger.info(f"filter_params_by_schema: fetching schema for {_SCHEMA_REPO}@{_SCHEMA_REF}")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            schema = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"{_SCHEMA_REPO}@{_SCHEMA_REF} not found.") from e
        ds.logger.warning(f"filter_params_by_schema: HTTP error fetching schema — {e}, skipping filter")
        return
    except Exception as e:
        ds.logger.warning(f"filter_params_by_schema: could not fetch schema — {e}, skipping filter")
        return

    allowed = set()
    for section in {**schema.get("$defs", {}), **schema.get("definitions", {})}.values():
        allowed.update(section.get("properties", {}).keys())

    ds.logger.info(f"filter_params_by_schema: schema defines {len(allowed):,} parameters")

    removed = [
        key
        for key in list(ds.params.keys())
        if key not in allowed and key not in _CIRRO_PASSTHROUGH_PARAMS
    ]
    for key in removed:
        ds.remove_param(key, force=True)

    if removed:
        ds.logger.info(f"filter_params_by_schema: removed {len(removed)} unrecognized param(s): {removed}")
    ds.logger.info(f"filter_params_by_schema: {len(ds.params)} param(s) remain")


if __name__ == "__main__":

    ds = PreprocessDataset.from_running()

    ds.logger.info(f"Starting sarek_align preprocess — {_SCHEMA_REPO}@{_SCHEMA_REF}")

    require_analysis_type_binding(ds)

    manifest = make_manifest(ds)
    ds.logger.info(manifest.to_csv(index=None))
    # Write to the dataset's config/ folder (mapped in process-input.json)
    manifest.to_csv(ds.params["input"], index=None)
    ds.logger.info(f"Wrote {manifest.shape[0]} row(s) to {ds.params['input']}")

    # Warn about custom-genome limitations while genome_source is still present.
    warn_custom_genome_limitations(ds)

    # Capture the genome source before resolve_reference_genome removes it.
    is_custom_genome = ds.params.get("genome_source") == "dataset"

    # Resolve the reference genome (iGenomes vs Custom BWA index)
    resolve_reference_genome(ds)

    # `compute_multiplier` == 2 for WGS and 1 for WES; consumed by process-compute.config
    wes = ds.params["wes"]
    compute_multiplier = int(2 - int(wes))
    ds.add_param("compute_multiplier", compute_multiplier)
    ds.logger.info(f"compute_multiplier={compute_multiplier} ({'WES' if wes else 'WGS'})")

    # If an intervals file was not selected, use --no_intervals
    if not ds.params.get("intervals"):
        ds.add_param("no_intervals", True)
        ds.logger.info("No intervals file selected — adding --no_intervals flag")

    apply_extra_json_params(ds)

    stage_colliding_vcf_params(ds)

    filter_params_by_schema(ds)

    # With all params populated, skip base recalibration for custom genomes that
    # lack known-sites resources (dbsnp/known_indels). iGenomes supplies these.
    skip_baserecalibration_without_known_sites(ds, is_custom_genome)

    ds.logger.info(f"Final params ({len(ds.params)} total): {ds.params}")
