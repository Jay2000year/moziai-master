from setuptools import (
    find_packages,
    setup,
)

# setup(
#     name="mozi_ai",
#     packages=find_packages(where=".", include="mozi*"),  # 必须包含__init__.py的文件夹
#     version="0.0.1",
#     author="hsfw",
#     description="墨子AI的初始pip安装版本",
#     long_description="墨子AI的初始pip安装版本0.0.1",
#     url="https://gitee.com/hs-defense/moziai",
#     install_requires=[
#         'absl-py>=0.1.0',
#         'deepdiff',
#         'enum34',
#         'future'],
#     classifiers=[
#         'Development Status :: 4 - Beta',  # 3 - Alpha；5 - Production/Stable
#         'Operating System :: Microsoft'  # 你的操作系统
#         'Intended Audience :: Developers',
#         'License :: OSI Approved :: BSD License',  # BSD认证
#         'Programming Language :: Python',  # 支持的语言
#         'Programming Language :: Python :: 3.6',
#         'Programming Language :: Python :: 3.9',
#         'Topic :: Software Development :: Libraries'
#     ],
#     zip_safe=True,  # 不压缩包，以目录的形式安装
#     # 安装过程中，需要安装的静态文件，如配置文件、service文件、图片等
#     data_files=[
#         ('', ['conf/*.conf']),
#         ('/usr/lib/systemd/system/', ['bin/*.service']),],
#
#     # 希望被打包的文件
#     package_data={
#         '':['*.txt'],
#         'bandwidth_reporter':['*.txt']},
#     # 不打包某些文件
#     exclude_package_data={
#         'bandwidth_reporter':['*.txt']
#                }
# )

with open(
    "README.md",
    encoding="utf-8",
) as f:
    readme = f.read()

with open(
    "LICENSE",
    encoding="utf-8",
) as f:
    licens = f.read()

with open(
    "requirements.txt",
    encoding="utf-8",
) as f:
    reqs = f.read()
pkgs = [
    p
    for p in find_packages()
    if p.startswith("mozi")
]

setup(
    name="mozi-ai",
    version="0.1.1.3",
    url="https://gitee.com/hs-defense/moziai",
    description="墨子AI:军事人工智能领航者, developed by HSFW",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License",
    author="hsfw",
    python_requires="==3.6.9",
    packages=pkgs,
    install_requires=reqs.strip().split("\n"),
    include_package_data=True,
    package_data={
        "": ["*"]
    },  # 包含了所有静态文件包括pt文件，txt，json等
)
