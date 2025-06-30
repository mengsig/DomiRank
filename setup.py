from setuptools import setup, find_packages

# read requirements.txt
with open("requirements.txt") as f:
    reqs = [r.strip() for r in f if r.strip() and not r.startswith("#")]

setup(
    name="domirank",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=reqs,        # ← install everything in requirements.txt
    include_package_data=True,    # if you have MANIFEST.in or data files
    zip_safe=False,
)
