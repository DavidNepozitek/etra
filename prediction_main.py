

if __name__ == '__main__':
    X, y, groups  = generate_dataset("data", ALL_SUBJECTS, ["Puzzle", "Waldo", "Natural"])

    clf = make_pipeline(
        TruncationTransformer(lower=250),
        TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False),
        RandomForestClassifier(),
    )

    # clf = ColumnEnsembleClassifier(
    #     estimators=[
    #         ("TSF0", TimeSeriesForestClassifier(n_estimators=100), ["LX"])
    #     ]
    # )
    # clf.fit(X["LX"].to_frame(), y)
    # print(clf.score(X["LX"].to_frame(), y))

    logo = LeaveOneGroupOut()
    scores = cross_val_score(clf, X["LX"].to_frame(), y, groups=groups, cv=logo)
    print(scores.mean())
