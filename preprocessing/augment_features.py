from openfe import OpenFE, tree_to_formula, transform


def augment_features(train_df, val_df, y="tow", n_jobs=4):
    train_y = train_df[y]
    train_x = train_df.drop(columns=[y])

    val_y = val_df[y]
    val_x = val_df.drop(columns=[y])

    ofe = OpenFE()
    ofe.fit(data=train_x, label=train_y, verbose=-1)

    # OpenFE recommends a list of new features. We include the top 10
    # generated features to see how they influence the model performance
    train_x, val_x = transform(
        train_x,
        val_x,
        ofe.new_features_list[:10],
    )

    for feature in ofe.new_features_list[:10]:
        print(tree_to_formula(feature))

    # add back the y column, we return the full datasets
    train_x[y] = train_y
    val_x[y] = val_y

    return train_x, val_x
