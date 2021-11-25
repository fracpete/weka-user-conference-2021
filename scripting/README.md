# Bridging the gap: Scripting Weka from Python

* Project pages:

  * https://github.com/fracpete/python-weka-wrapper3

  * https://github.com/fracpete/sklearn-weka-plugin

* Presentation: 

  * [.odp](scripting.odp)
  * [.pdf](scripting.pdf)

* Virtual environment for scripts

  ```
  virtualenv -p /usr/bin/python3 venv
  ./venv/bin/pip install numpy
  ./venv/bin/pip install javabridge
  ./venv/bin/pip install matplotlib pillow pygraphviz
  ./venv/bin/pip install python-weka-wrapper3
  ./venv/bin/pip install sklearn-weka-plugin
  ```

  Output of `pip freeze`:

  ```
  cycler==0.11.0
  fonttools==4.28.2
  javabridge==1.0.19
  joblib==1.1.0
  kiwisolver==1.3.2
  matplotlib==3.5.0
  numpy==1.21.4
  packaging==21.3
  Pillow==8.4.0
  pygraphviz==1.7
  pyparsing==3.0.6
  python-dateutil==2.8.2
  python-weka-wrapper3==0.2.4
  scikit-learn==1.0.1
  scipy==1.7.3
  setuptools-scm==6.3.2
  six==1.16.0
  sklearn==0.0
  sklearn-weka-plugin==0.0.3
  threadpoolctl==3.0.0
  tomli==1.2.2
  ```
