# DeepTactics-Muzero

<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/CogitoNTNU/DeepTactics-Muzero/frontend.yml)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/CogitoNTNU/DeepTactics-Muzero/backend.yml)
![GitHub top language](https://img.shields.io/github/languages/top/CogitoNTNU/DeepTactics-Muzero)
![GitHub language count](https://img.shields.io/github/languages/count/CogitoNTNU/DeepTactics-Muzero)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

<img src="docs/images/cogito-ntnu-deeptactics-logo.png" width="50%" alt="Cogito Project Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>


<details> 
<summary><b>📋 Table of contents </b></summary>

- [DeepTactics-Muzero](#deeptactics-muzero)
  - [Description](#description)
  - [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [Testing](#testing)
  - [Team](#team)
    - [License](#license)

</details>

## Description 
<!-- TODO: Provide a brief overview of what this project does and its key features. Please add pictures or videos of the application -->


## Getting started
<!-- TODO: In this Section you describe how to install this project in its intended environment.(i.e. how to get it to run)  
-->

<!-- TODO: Describe how to configure the project (environment variables, config files, etc.).

### Configuration
Create a `.env` file in the root directory of the project and add the following environment variables:

```bash
OPENAI_API_KEY = 'your_openai_api_key'
MONGODB_URI = 'your_secret_key'
```
-->
Use type hinting in all methods when contributing.
Here is an example: 
```Python
import numpy as np
import torch

def dummy_fun(a: int, b: ndarray, c: torch.tensor) -> list[int]:
  pass
```

### Prerequisites
<!-- TODO: In this section you put what is needed for the program to run.
For example: OS version, programs, libraries, etc.  

-->
- Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)

## Usage

To run the project, run the following command from the root directory of the project:

```bash
docker compose up --build
```
<!-- TODO: Instructions on how to run the project and use its features. -->

## Testing

To run the test suite, run the following command from the root directory of the project:

```bash
docker compose run backend python -m pytest
```

## Team

This project would not have been possible without the hard work and dedication of all of the contributors. Thank you for the time and effort you have put into making this project a reality.

<table align="center">
    <tr>
        <!--
        <td align="center">
            <a href="https://github.com/NAME_OF_MEMBER">
              <img src="https://github.com/NAME_OF_MEMBER.png?size=100" width="100px;" alt="NAME OF MEMBER"/><br />
              <sub><b>NAME OF MEMBER</b></sub>
            </a>
        </td>
        -->
    </tr>
</table>

![Group picture](docs/img/team.png)

### License

------
Distributed under the MIT License. See `LICENSE` for more information.
