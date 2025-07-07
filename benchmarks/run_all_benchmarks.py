#!/usr/bin/env python3
"""
Run all PuffinFlow benchmarks and generate comprehensive report.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class BenchmarkSuite:
    """Comprehensive benchmark suite runner."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.timestamp = datetime.now().isoformat()
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for benchmark context."""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cpu_freq_mhz": psutil.cpu_freq().current
            if psutil.cpu_freq()
            else "Unknown",
            "timestamp": self.timestamp,
        }

    def run_benchmark_module(
        self, module_name: str, module_path: str
    ) -> dict[str, Any]:
        """Run a single benchmark module."""
        print(f"\n{'='*80}")
        print(f"Running {module_name}")
        print(f"{'='*80}")

        start_time = time.time()

        try:
            # Load and run the benchmark module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Run the main function
            results = module.main()

            end_time = time.time()

            benchmark_info = {
                "module": module_name,
                "status": "success",
                "execution_time_seconds": round(end_time - start_time, 2),
                "results": [
                    {
                        "name": r.name,
                        "duration_ms": r.duration_ms,
                        "memory_mb": r.memory_mb,
                        "cpu_percent": r.cpu_percent,
                        "iterations": r.iterations,
                        "min_time": r.min_time,
                        "max_time": r.max_time,
                        "median_time": r.median_time,
                        "std_dev": r.std_dev,
                        "throughput_ops_per_sec": getattr(
                            r, "throughput_ops_per_sec", 0
                        ),
                    }
                    for r in results
                ],
            }

            print(
                f"\n✅ {module_name} completed successfully in {benchmark_info['execution_time_seconds']:.2f}s"
            )

        except Exception as e:
            end_time = time.time()
            benchmark_info = {
                "module": module_name,
                "status": "failed",
                "execution_time_seconds": round(end_time - start_time, 2),
                "error": str(e),
                "results": [],
            }

            print(f"\n❌ {module_name} failed: {e!s}")

        return benchmark_info

    def run_all_benchmarks(self) -> dict[str, Any]:
        """Run all benchmark modules."""
        benchmark_modules = [
            ("Core Agent Benchmarks", "benchmark_core_agent.py"),
            ("Resource Management Benchmarks", "benchmark_resource_management.py"),
            ("Coordination Benchmarks", "benchmark_coordination.py"),
            ("Observability Benchmarks", "benchmark_observability.py"),
        ]

        print("🚀 Starting PuffinFlow Comprehensive Benchmark Suite")
        print(f"Timestamp: {self.timestamp}")
        print(f"System Info: {self.system_info}")

        benchmark_results = []

        for module_name, module_file in benchmark_modules:
            module_path = Path(__file__).parent / module_file

            if module_path.exists():
                result = self.run_benchmark_module(module_name, module_path)
                benchmark_results.append(result)
            else:
                print(f"⚠️  Warning: {module_file} not found, skipping...")
                benchmark_results.append(
                    {
                        "module": module_name,
                        "status": "skipped",
                        "execution_time_seconds": 0,
                        "error": "Module file not found",
                        "results": [],
                    }
                )

        # Compile final results
        final_results = {
            "system_info": self.system_info,
            "benchmark_run": {
                "timestamp": self.timestamp,
                "total_modules": len(benchmark_modules),
                "successful_modules": len(
                    [r for r in benchmark_results if r["status"] == "success"]
                ),
                "failed_modules": len(
                    [r for r in benchmark_results if r["status"] == "failed"]
                ),
                "skipped_modules": len(
                    [r for r in benchmark_results if r["status"] == "skipped"]
                ),
                "total_execution_time_seconds": sum(
                    r["execution_time_seconds"] for r in benchmark_results
                ),
            },
            "benchmarks": benchmark_results,
        }

        self.results = final_results
        return final_results

    def save_results(self, format: str = "json") -> str:
        """Save benchmark results to file."""
        if format == "json":
            filename = (
                f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            filepath = self.output_dir / filename

            with filepath.open("w") as f:
                json.dump(self.results, f, indent=2)

            return str(filepath)

        elif format == "html":
            filename = (
                f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            filepath = self.output_dir / filename

            html_content = self._generate_html_report()

            with filepath.open("w") as f:
                f.write(html_content)

            return str(filepath)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_html_report(self) -> str:
        """Generate HTML report from benchmark results."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>PuffinFlow Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .benchmark-section { margin: 20px 0; }
        .benchmark-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        .benchmark-table th, .benchmark-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .benchmark-table th { background-color: #f2f2f2; }
        .success { color: green; }
        .failed { color: red; }
        .skipped { color: orange; }
        .summary { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>PuffinFlow Benchmark Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>System:</strong> {platform} | Python {python_version} | CPU: {cpu_count} cores | Memory: {memory_total_gb} GB</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <ul>
            <li><strong>Total Modules:</strong> {total_modules}</li>
            <li><strong>Successful:</strong> <span class="success">{successful_modules}</span></li>
            <li><strong>Failed:</strong> <span class="failed">{failed_modules}</span></li>
            <li><strong>Skipped:</strong> <span class="skipped">{skipped_modules}</span></li>
            <li><strong>Total Execution Time:</strong> {total_execution_time_seconds:.2f} seconds</li>
        </ul>
    </div>

    {benchmark_sections}
</body>
</html>
        """

        # Generate benchmark sections
        benchmark_sections = ""

        for benchmark in self.results["benchmarks"]:
            status_class = benchmark["status"]

            section_html = f"""
    <div class="benchmark-section">
        <h2>{benchmark['module']} <span class="{status_class}">({benchmark['status']})</span></h2>
        <p><strong>Execution Time:</strong> {benchmark['execution_time_seconds']:.2f} seconds</p>

        {self._generate_benchmark_table(benchmark)}
    </div>
            """

            benchmark_sections += section_html

        return html_template.format(
            timestamp=self.results["system_info"]["timestamp"],
            platform=self.results["system_info"]["platform"],
            python_version=self.results["system_info"]["python_version"].split()[0],
            cpu_count=self.results["system_info"]["cpu_count"],
            memory_total_gb=self.results["system_info"]["memory_total_gb"],
            total_modules=self.results["benchmark_run"]["total_modules"],
            successful_modules=self.results["benchmark_run"]["successful_modules"],
            failed_modules=self.results["benchmark_run"]["failed_modules"],
            skipped_modules=self.results["benchmark_run"]["skipped_modules"],
            total_execution_time_seconds=self.results["benchmark_run"][
                "total_execution_time_seconds"
            ],
            benchmark_sections=benchmark_sections,
        )

    def _generate_benchmark_table(self, benchmark: dict[str, Any]) -> str:
        """Generate HTML table for benchmark results."""
        if benchmark["status"] != "success" or not benchmark["results"]:
            if benchmark["status"] == "failed":
                return f'<p class="failed">Error: {benchmark.get("error", "Unknown error")}</p>'
            return "<p>No results available</p>"

        table_html = """
        <table class="benchmark-table">
            <tr>
                <th>Benchmark</th>
                <th>Avg (ms)</th>
                <th>Min (ms)</th>
                <th>Max (ms)</th>
                <th>Median (ms)</th>
                <th>Std Dev</th>
                <th>Throughput (ops/s)</th>
                <th>Memory (MB)</th>
                <th>Iterations</th>
            </tr>
        """

        for result in benchmark["results"]:
            table_html += f"""
            <tr>
                <td>{result['name']}</td>
                <td>{result['duration_ms']:.2f}</td>
                <td>{result['min_time']:.2f}</td>
                <td>{result['max_time']:.2f}</td>
                <td>{result['median_time']:.2f}</td>
                <td>{result['std_dev']:.2f}</td>
                <td>{result['throughput_ops_per_sec']:.2f}</td>
                <td>{result['memory_mb']:.2f}</td>
                <td>{result['iterations']}</td>
            </tr>
            """

        table_html += "</table>"
        return table_html

    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 100)
        print("🎯 BENCHMARK SUITE SUMMARY")
        print("=" * 100)

        print(f"📊 Total Modules: {self.results['benchmark_run']['total_modules']}")
        print(f"✅ Successful: {self.results['benchmark_run']['successful_modules']}")
        print(f"❌ Failed: {self.results['benchmark_run']['failed_modules']}")
        print(f"⏭️  Skipped: {self.results['benchmark_run']['skipped_modules']}")
        print(
            f"⏱️  Total Time: {self.results['benchmark_run']['total_execution_time_seconds']:.2f}s"
        )

        # Print top performing benchmarks
        all_results = []
        for benchmark in self.results["benchmarks"]:
            if benchmark["status"] == "success":
                for result in benchmark["results"]:
                    all_results.append(
                        {
                            "module": benchmark["module"],
                            "name": result["name"],
                            "throughput": result["throughput_ops_per_sec"],
                            "duration": result["duration_ms"],
                        }
                    )

        if all_results:
            print("\n🏆 TOP PERFORMING BENCHMARKS (by throughput):")
            top_benchmarks = sorted(
                all_results, key=lambda x: x["throughput"], reverse=True
            )[:10]

            for i, benchmark in enumerate(top_benchmarks, 1):
                print(
                    f"{i:2d}. {benchmark['name']:<40} | {benchmark['throughput']:>10.2f} ops/s | {benchmark['duration']:>8.2f}ms"
                )

            print("\n⚡ FASTEST BENCHMARKS (by duration):")
            fastest_benchmarks = sorted(all_results, key=lambda x: x["duration"])[:10]

            for i, benchmark in enumerate(fastest_benchmarks, 1):
                print(
                    f"{i:2d}. {benchmark['name']:<40} | {benchmark['duration']:>8.2f}ms | {benchmark['throughput']:>10.2f} ops/s"
                )

        print("=" * 100)


def main():
    """Main function to run benchmark suite."""
    parser = argparse.ArgumentParser(description="Run PuffinFlow benchmark suite")
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument(
        "--format", choices=["json", "html"], default="json", help="Output format"
    )
    parser.add_argument(
        "--save-results", action="store_true", help="Save results to file"
    )

    args = parser.parse_args()

    # Create and run benchmark suite
    suite = BenchmarkSuite(output_dir=args.output_dir)

    # Run all benchmarks
    results = suite.run_all_benchmarks()

    # Print summary
    suite.print_summary()

    # Save results if requested
    if args.save_results:
        filepath = suite.save_results(format=args.format)
        print(f"\n💾 Results saved to: {filepath}")

    # Return exit code based on results
    if results["benchmark_run"]["failed_modules"] > 0:
        print(
            f"\n⚠️  {results['benchmark_run']['failed_modules']} benchmark modules failed!"
        )
        return 1

    print("\n🎉 All benchmarks completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
