"""Validation shared by split previews and full dataset preparation."""


def excluded_groups(policy, assigned_groups):
    groups = policy.get("exclude_groups", [])
    if (not isinstance(groups, list)
            or any(not isinstance(group, str) or not group.strip() for group in groups)):
        raise ValueError("exclude_groups must be a list of nonempty family names")
    if len(groups) != len(set(groups)):
        raise ValueError("exclude_groups must not contain duplicate families")
    overlap = set(groups) & set(assigned_groups)
    if overlap:
        raise ValueError(f"Excluded families also assigned to a partition: {sorted(overlap)}")
    return set(groups)
