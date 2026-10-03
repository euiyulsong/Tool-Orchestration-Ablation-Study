    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-dir", type=Path, default=Path("bfcl_cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("ablation_4way_outputs"))
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--start", type=int, default=1, help="1-based sample start")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    samples = load_bfcl_samples(
        dataset_dir=args.dataset_dir,
        n=args.n,
        seed=args.seed,
    )

    with (args.out_dir / "samples.json").open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    records = []

    jsonl_path = args.out_dir / "results.jsonl"

    # Resume support: read existing successful sample IDs.
    done = set()
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    old = json.loads(line)
                    sid = old["sample"]["sample_id"]
                    result_map = old.get("results", {})
                    all_ok = (
                        all(
                            result_map.get(s, {}).get("ok") is True
                            for s in "ABCD"
                        )
                    )
                    # Only successful 4-way samples are resumable.
                    # Failed samples are rerun after code/schema fixes.
                    if all_ok:
                        done.add(sid)
                        records.append(old)
                except Exception:
                    pass

    for sample in samples:
        sid = sample["sample_id"]
        if sid < args.start or sid in done:
            continue

        print("\n" + "=" * 100)
        print(f"[{sid:02d}/{len(samples)}] {sample['category']} | {sample['bfcl_id']}")
        print(sample["query"])
        print("tools:", ", ".join(t["name"] for t in sample["tools"]))

        result_map = {}

        for label, runner in [
            ("A", run_A),
            ("B", run_B),
            ("C", run_C),
            ("D", run_D),
        ]:
            print(f"  -> {label}", end="", flush=True)
            try:
                r = runner(client, sample)
                result_map[label] = r
                print(
                    f"  ok={r.get('ok')} "
                    f"api={r.get('api_calls')} "
                    f"lat={r.get('latency_total_s',0):.2f}s "
                    f"tokens={total_tokens_for_result(r)}"
                )
            except Exception as e:
                traceback.print_exc()
                result_map[label] = {
                    "strategy": label,
                    "name": "exception",
                    "ok": False,
                    "error": repr(e),
                    "pre_text": "",
                    "calls": [],
                    "observations": [],
                    "next_action": "",
                    "latency_first_s": 0.0,
                    "latency_second_s": 0.0,
                    "latency_total_s": 0.0,
                    "api_calls": 0,
                    "usage_first": {},
                    "usage_second": {},
                }

        rec = {
            "sample": sample,
            "results": result_map,
        }

        if args.judge:
            try:
                rec["judge"] = judge_four(client, sample, result_map)
            except Exception as e:
                rec["judge"] = {"error": repr(e)}

        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        records.append(rec)

    # Deduplicate if resumed.
    by_id = {}
    for r in records:
        by_id[r["sample"]["sample_id"]] = r
    records = [by_id[k] for k in sorted(by_id)]

    # CSV
    flat = [flatten_row(r) for r in records]
    csv_path = args.out_dir / "results.csv"
    if flat:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)

    # Summary CSV
    summary = summarize(records)
    summary_path = args.out_dir / "summary.csv"
    if summary:
        with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    # PNG
    render_png_pages(records, args.out_dir, per_page=5)

    print("\nDONE")
    print("samples :", args.out_dir / "samples.json")
    print("jsonl   :", jsonl_path)
    print("csv     :", csv_path)
    print("summary :", summary_path)
    print("png dir :", args.out_dir / "png")


if __name__ == "__main__":
    main()
