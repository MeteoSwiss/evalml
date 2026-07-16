# Configuring CI/CD

Details about using CI/CD in CSCS can be found at https://docs.cscs.ch/services/cicd/.

This was set up using the checklist here: https://docs.cscs.ch/services/cicd/#enable-ci-for-your-project

## Tests run using CI/CD

The configuration `cscs.yml` controls which tests are run. For example,
`unit_test_job` calls the command `pytest tests/unit` (after setting up an environment).
Adding another test suite is relatively simple; just make a new job.

These tests should be triggered automatically in PRs. 

## Admin setup
The CI project `mch-evalml` controls how the 

The CI administrative setup is available to "trusted users" which are specified
in the administrative interface. As of 14.7.2026, these users are `mmcgloho`, `clairemerker`, and `frazane`.

## Troubleshooting

### Disabling and enabling CI/CD
To disable CI/CD (for example, if balfrin is down, or there is some other problem),
navigate to the [administrative interface](https://cicd-ext-mw.cscs.ch/ci/setup/ui?repo=6067442399726097), go to `Default CI enabled branches`, and remove `main` (or any other branches you wish to disable).

To enable, just add it back.

### Tests aren't running.
