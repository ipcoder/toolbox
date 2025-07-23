from inu.env import EnvLoc
from toolbox.vis.annotations import IssueCollection
from pathlib import Path

def test_issues():

    issues = IssueCollection.from_csv(EnvLoc.ISSUES.first_existing() / 'selector_issues.csv')
    orig_issues = issues.copy()
    issue = {'issue_type': '1',  'alg': 'alg_1', 'id': 0, 'scene': 'A', 'dataset': 'FT3D',
             'polygon': ((1,2), (2,3), (3,4))}

    issues.add(issue)
    assert len(orig_issues) < len(issues)

    issues.remove(issue)
    assert len(orig_issues) == len(issues)

