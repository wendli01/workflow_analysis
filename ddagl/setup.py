import setuptools

requirements = ["dgl-cu110==0.5.3",
                "networkx>=2.4",
                "pandas>=1.1.5",
                "torch==1.7.1",
                'scikit-learn>=0.23.2',
                'scipy>=1.5.2',
                "seaborn>=0.11.0",
                "numpy>=1.18.2",
                "pip<=18"]

setuptools.setup(name='ddagl', packages=['ddagl'], version="0.1",
                 author='Lorenz Wendlinger', author_email='lorenz.wendlinger@uni-passau.de',
                 url="https://github.com/wendli01/workflow_analysis",
                 description="Deep Directed Acyclical Graph Learning tools",
                 long_description="Tools for unsupervised and supervised learning on DAGs",
                 install_requires=requirements, python_requires=">=3.7",)
