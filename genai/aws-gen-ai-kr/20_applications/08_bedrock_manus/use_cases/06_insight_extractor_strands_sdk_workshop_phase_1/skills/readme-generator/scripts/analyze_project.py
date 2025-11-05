#!/usr/bin/env python3
"""
Project Structure Analyzer for README Generation

This script automatically analyzes a project's codebase to extract information
useful for generating README.md files.

Usage:
    python analyze_project.py <project_root_path>

Output:
    JSON structure with project metadata including:
    - Entry point files
    - Dependencies and frameworks
    - Directory structure
    - Existing documentation
    - Configuration files
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class ProjectAnalyzer:
    """Analyzes project structure and extracts metadata for README generation."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {project_root}")

        self.analysis_result = {
            "project_root": str(self.project_root),
            "project_name": self.project_root.name,
            "entry_points": [],
            "dependencies": {},
            "frameworks": [],
            "directory_structure": {},
            "documentation_files": [],
            "configuration_files": [],
            "test_directories": [],
            "license_file": None,
            "readme_exists": False,
            "python_version": None,
        }

    def analyze(self) -> Dict[str, Any]:
        """Run complete project analysis."""
        print(f"Analyzing project: {self.project_root}")

        self._find_entry_points()
        self._analyze_dependencies()
        self._detect_frameworks()
        self._map_directory_structure()
        self._find_documentation()
        self._find_configuration_files()
        self._find_test_directories()
        self._find_license()
        self._check_readme_exists()

        return self.analysis_result

    def _find_entry_points(self):
        """Identify main entry point files."""
        entry_point_patterns = [
            "main.py",
            "app.py",
            "run.py",
            "server.py",
            "cli.py",
            "__main__.py",
            "manage.py",  # Django
        ]

        for pattern in entry_point_patterns:
            matches = list(self.project_root.rglob(pattern))
            # Exclude virtual environments and build directories
            matches = [m for m in matches if not self._is_excluded_path(m)]

            for match in matches:
                rel_path = match.relative_to(self.project_root)
                self.analysis_result["entry_points"].append({
                    "file": str(rel_path),
                    "type": pattern,
                    "absolute_path": str(match)
                })

        print(f"  Found {len(self.analysis_result['entry_points'])} entry points")

    def _analyze_dependencies(self):
        """Extract dependencies from various dependency files."""
        dependency_files = {
            "pyproject.toml": self._parse_pyproject_toml,
            "requirements.txt": self._parse_requirements_txt,
            "setup.py": self._parse_setup_py,
            "Pipfile": self._parse_pipfile,
            "environment.yml": self._parse_environment_yml,
            "package.json": self._parse_package_json,
        }

        for filename, parser in dependency_files.items():
            file_path = self.project_root / filename
            if file_path.exists():
                try:
                    deps = parser(file_path)
                    self.analysis_result["dependencies"][filename] = deps
                except Exception as e:
                    print(f"  Warning: Failed to parse {filename}: {e}")

        dep_count = sum(len(deps) for deps in self.analysis_result["dependencies"].values())
        print(f"  Found {dep_count} dependencies across {len(self.analysis_result['dependencies'])} files")

    def _parse_pyproject_toml(self, file_path: Path) -> List[str]:
        """Parse pyproject.toml dependencies."""
        try:
            import tomli
        except ImportError:
            # Fallback to simple regex parsing
            content = file_path.read_text()
            deps = re.findall(r'"([a-zA-Z0-9_-]+)(?:[>=<~!].*?)?"', content)
            return list(set(deps))

        with open(file_path, "rb") as f:
            data = tomli.load(f)

        deps = []

        # Check different sections
        if "project" in data and "dependencies" in data["project"]:
            deps.extend(data["project"]["dependencies"])

        if "tool" in data and "poetry" in data["tool"] and "dependencies" in data["tool"]["poetry"]:
            deps.extend(data["tool"]["poetry"]["dependencies"].keys())

        # Extract Python version
        if "project" in data and "requires-python" in data["project"]:
            self.analysis_result["python_version"] = data["project"]["requires-python"]

        return [self._clean_dependency_name(dep) for dep in deps]

    def _parse_requirements_txt(self, file_path: Path) -> List[str]:
        """Parse requirements.txt dependencies."""
        content = file_path.read_text()
        deps = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(self._clean_dependency_name(line))
        return deps

    def _parse_setup_py(self, file_path: Path) -> List[str]:
        """Parse setup.py dependencies."""
        content = file_path.read_text()

        # Extract from install_requires
        install_requires_match = re.search(
            r'install_requires\s*=\s*\[(.*?)\]',
            content,
            re.DOTALL
        )

        if install_requires_match:
            deps_str = install_requires_match.group(1)
            deps = re.findall(r'["\']([^"\']+)["\']', deps_str)
            return [self._clean_dependency_name(dep) for dep in deps]

        return []

    def _parse_pipfile(self, file_path: Path) -> List[str]:
        """Parse Pipfile dependencies."""
        try:
            import tomli
            with open(file_path, "rb") as f:
                data = tomli.load(f)

            deps = []
            if "packages" in data:
                deps.extend(data["packages"].keys())
            if "dev-packages" in data:
                deps.extend(data["dev-packages"].keys())

            return deps
        except:
            return []

    def _parse_environment_yml(self, file_path: Path) -> List[str]:
        """Parse conda environment.yml dependencies."""
        try:
            import yaml
            with open(file_path) as f:
                data = yaml.safe_load(f)

            deps = []
            if "dependencies" in data:
                for dep in data["dependencies"]:
                    if isinstance(dep, str):
                        deps.append(self._clean_dependency_name(dep))
                    elif isinstance(dep, dict) and "pip" in dep:
                        deps.extend([self._clean_dependency_name(d) for d in dep["pip"]])

            return deps
        except:
            return []

    def _parse_package_json(self, file_path: Path) -> List[str]:
        """Parse package.json dependencies (for JS/TS projects)."""
        try:
            data = json.loads(file_path.read_text())
            deps = []
            if "dependencies" in data:
                deps.extend(data["dependencies"].keys())
            if "devDependencies" in data:
                deps.extend(data["devDependencies"].keys())
            return deps
        except:
            return []

    def _clean_dependency_name(self, dep: str) -> str:
        """Extract clean package name from dependency string."""
        # Remove version specifiers
        dep = re.split(r'[>=<~!]', dep)[0]
        # Remove extras
        dep = re.split(r'\[', dep)[0]
        return dep.strip()

    def _detect_frameworks(self):
        """Detect popular frameworks used in the project."""
        framework_indicators = {
            "Django": ["django", "manage.py"],
            "Flask": ["flask"],
            "FastAPI": ["fastapi"],
            "Streamlit": ["streamlit"],
            "LangChain": ["langchain"],
            "Strands SDK": ["strands-agents", "strands"],
            "LangGraph": ["langgraph"],
            "Boto3": ["boto3"],
            "TensorFlow": ["tensorflow"],
            "PyTorch": ["torch", "pytorch"],
            "Scikit-learn": ["scikit-learn", "sklearn"],
            "Pandas": ["pandas"],
            "NumPy": ["numpy"],
            "Jupyter": ["jupyter", "ipynb"],
        }

        all_deps = []
        for deps in self.analysis_result["dependencies"].values():
            all_deps.extend([d.lower() for d in deps])

        detected_frameworks = []
        for framework, indicators in framework_indicators.items():
            for indicator in indicators:
                if any(indicator.lower() in dep for dep in all_deps):
                    detected_frameworks.append(framework)
                    break
                # Also check for actual files
                if indicator.endswith(".py"):
                    if list(self.project_root.rglob(indicator)):
                        detected_frameworks.append(framework)
                        break

        self.analysis_result["frameworks"] = list(set(detected_frameworks))
        print(f"  Detected frameworks: {', '.join(self.analysis_result['frameworks'])}")

    def _map_directory_structure(self):
        """Map high-level directory structure."""
        important_dirs = [
            "src",
            "lib",
            "app",
            "tests",
            "docs",
            "examples",
            "scripts",
            "setup",
            "config",
            "data",
            "models",
            "notebooks",
            "artifacts",
            "output",
            "static",
            "templates",
        ]

        structure = {}
        for dir_name in important_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                # Count files and subdirectories
                items = list(dir_path.iterdir())
                structure[dir_name] = {
                    "exists": True,
                    "file_count": len([i for i in items if i.is_file()]),
                    "dir_count": len([i for i in items if i.is_dir()]),
                }

        self.analysis_result["directory_structure"] = structure
        print(f"  Mapped {len(structure)} important directories")

    def _find_documentation(self):
        """Find existing documentation files."""
        doc_patterns = [
            "README.md",
            "README.rst",
            "README.txt",
            "CLAUDE.md",
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            "HISTORY.md",
            "docs/*.md",
            "docs/*.rst",
        ]

        docs = []
        for pattern in doc_patterns:
            matches = list(self.project_root.glob(pattern))
            for match in matches:
                if match.is_file():
                    rel_path = match.relative_to(self.project_root)
                    docs.append({
                        "file": str(rel_path),
                        "size_bytes": match.stat().st_size,
                    })

        self.analysis_result["documentation_files"] = docs
        print(f"  Found {len(docs)} documentation files")

    def _find_configuration_files(self):
        """Find configuration files."""
        config_patterns = [
            ".env.example",
            ".env.template",
            "config.yml",
            "config.yaml",
            "config.json",
            "settings.py",
            ".gitignore",
            ".dockerignore",
            "Dockerfile",
            "docker-compose.yml",
        ]

        configs = []
        for pattern in config_patterns:
            file_path = self.project_root / pattern
            if file_path.exists():
                rel_path = file_path.relative_to(self.project_root)
                configs.append(str(rel_path))

        self.analysis_result["configuration_files"] = configs
        print(f"  Found {len(configs)} configuration files")

    def _find_test_directories(self):
        """Find test directories."""
        test_patterns = ["tests", "test", "testing"]

        test_dirs = []
        for pattern in test_patterns:
            dir_path = self.project_root / pattern
            if dir_path.exists() and dir_path.is_dir():
                test_dirs.append(pattern)

        self.analysis_result["test_directories"] = test_dirs
        print(f"  Found {len(test_dirs)} test directories")

    def _find_license(self):
        """Find license file."""
        license_patterns = ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"]

        for pattern in license_patterns:
            file_path = self.project_root / pattern
            if file_path.exists():
                self.analysis_result["license_file"] = pattern
                # Try to detect license type
                content = file_path.read_text()[:500].upper()
                if "MIT" in content:
                    self.analysis_result["license_type"] = "MIT"
                elif "APACHE" in content:
                    self.analysis_result["license_type"] = "Apache"
                elif "GPL" in content:
                    self.analysis_result["license_type"] = "GPL"
                break

        if self.analysis_result["license_file"]:
            print(f"  Found license: {self.analysis_result.get('license_type', 'Unknown')}")

    def _check_readme_exists(self):
        """Check if README already exists."""
        readme_files = ["README.md", "README.rst", "README.txt"]
        for readme in readme_files:
            if (self.project_root / readme).exists():
                self.analysis_result["readme_exists"] = True
                self.analysis_result["existing_readme"] = readme
                break

    def _is_excluded_path(self, path: Path) -> bool:
        """Check if path should be excluded from analysis."""
        excluded_parts = [
            ".venv",
            "venv",
            "env",
            ".env",
            "node_modules",
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".tox",
            "build",
            "dist",
            "*.egg-info",
        ]

        path_str = str(path)
        return any(excluded in path_str for excluded in excluded_parts)


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_project.py <project_root_path>")
        sys.exit(1)

    project_root = sys.argv[1]

    try:
        analyzer = ProjectAnalyzer(project_root)
        result = analyzer.analyze()

        # Pretty print JSON result
        print("\n" + "="*60)
        print("PROJECT ANALYSIS RESULT")
        print("="*60 + "\n")
        print(json.dumps(result, indent=2))

        # Save to file
        output_file = Path(project_root) / "project_analysis.json"
        output_file.write_text(json.dumps(result, indent=2))
        print(f"\n✓ Analysis saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
