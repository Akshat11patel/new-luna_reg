        print(f"FMT scale ratio        : {fmt_scale:.4f}")
        print(
            f"FMT rotation estimate  : "
            f"{fmt_rotation:.2f} degrees"
        )
        print(f"FMT confidence         : {fmt_confidence:.4f}")

        if fmt_confidence >= proc.FMT_STRONG_CONFIDENCE:
            print(
                "FMT decision           : "
                "STRONG ENOUGH for safe scale assistance"
            )
        else:
            print("FMT decision           : DIAGNOSTIC ONLY")

        moving, fmt_normalization = (
            proc.apply_fmt_scale_normalization(
                reference,
                moving,
                fmt_scale,
                fmt_confidence,
            )
        )

        scale_info["fmt"] = {
            "estimated_scale_ratio": fmt_scale,
            "estimated_rotation_deg": fmt_rotation,
            "confidence": fmt_confidence,
            **fmt_normalization,
        }

        print(
            "FMT scale applied      :",
            fmt_normalization["fmt_scale_applied"],
        )
        print("Target after FMT shape :", moving.shape)

        print("\n[4] Image-only illumination analysis")
        illum = proc.illumination_analysis(
            reference,
            moving,
            None,
            None,
        )

        for key, value in illum.items():
            print(f"{key}: {value}")

        mode_name = "WITHOUT_METADATA"

    # ============================================================
    # COMMON REGISTRATION PIPELINE
    # ============================================================
    out.save_working_images(run_dir, reference, moving)

    print(
        "\n[7] Feature method"
        if use_metadata
        else "\n[4] Feature method"
    )
    feature_method = choose_feature_method()
    print("Selected feature method:", feature_method)

    print(
        "\n[8] Registration"
        if use_metadata
        else "\n[5] Registration"
    )

    result, attempts = proc.adaptive_registration(
        reference,
        moving,
        illum,
        feature_method,
    )

    # All branch/intermediate images are written only by output.py.
    out.save_attempt_diagnostics(
        run_dir,
        reference,
        moving,
        attempts,
    )

    out.print_final_quality(
        result,
        mode_name,
        role_selection,
    )

    validation_summary = out.print_validation_summary(result)

    out.save_final_outputs(
        run_dir=run_dir,
        reference=reference,
        moving=moving,
        result=result,
        illumination=illum,
        scale_info=scale_info,
        reference_metadata=reference_metadata,
        moving_metadata=moving_metadata,
        overlap_info=overlap_info,
        input_mode=mode_name,
        metadata_used=use_metadata,
        role_selection=role_selection,
        validation_summary=validation_summary,
    )

    print("\nAttempt summary:")
    for attempt_result, _, _ in attempts:
        print(
            f"  {attempt_result['branch']}: "
            f"inliers={attempt_result['inliers']}, "
            f"ratio={attempt_result['inlier_ratio']:.2f}%, "
            f"error="
            f"{attempt_result['mean_reprojection_error']:.3f}px, "
            f"coverage="
            f"{attempt_result['spatial_coverage']:.2f}%, "
            f"pass={proc.quality_pass(attempt_result)}"
        )

    print("\nAll files for this run are stored in:")
    print(run_dir)


if __name__ == "__main__":
    main()
