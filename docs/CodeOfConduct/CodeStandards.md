# Things to do for developers

## Set up a virtual environment

A virtual environment is a self-contained directory that contains a Python installation for a particular version of Python, as well as a number of additional packages. This allows you to work on multiple projects with different dependencies without worrying about conflicts between them.

Creating a virtual environment depends on the OS you are using. For example, on Linux, you can use the following command:

```bash
python3 -m venv myenv
```

and on Windows, you can use the following command:

```bash
python -m venv myenv
```

This will create a directory called `myenv` that contains a Python installation and a number of additional packages. You can activate the virtual environment by running the following command:

```bash
source myenv/bin/activate
```

This will activate the virtual environment and you can install packages using `pip` without affecting the global Python installation.

## Use type hinting when writing code

Type hinting is a way to specify the type of a variable in Python. This is useful for developers to understand the expected type of a variable when reading code. It also helps to catch type errors early in the development process.
Instead of writing code like this:

```python
def add(a, b):
    return a + b
```

You must write code like this:

```python
def add(a: int, b: int) -> int:
    return a + b
```

This way, developers can understand that the function `add` expects two integers as input and returns an integer as output.

## CREATE tests for the code you produce

Writing tests is an important part of the development process. It helps to ensure that the code you write works as expected and catches bugs early in the development process. You should write tests for all the code you produce, including functions, classes, and modules.

Especially for large AI algorithms, this is key, since the code will run fine even if you have an error in the calculations.
The problem is that you will see training runs which aren't going anywhere, and you will have to debug the code to find the error.
If we have tests from the beginning, we can catch these errors early in the development process.

For more information on tests, see the [create tests](create-tests.md) document.
