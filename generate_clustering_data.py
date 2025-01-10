import pandas as pd
import numpy as np


def load_data():
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "CustomerID": range(1, 301),
            "Age": np.concatenate(
                [
                    np.random.randint(60, 80, 50),  # Luxury Retirees
                    np.random.randint(30, 50, 50),  # Active Gamers
                    np.random.randint(18, 25, 50),  # Urban Sport Enthusiasts
                    np.random.randint(30, 40, 50),  # Gourmet Gardeners
                    np.random.randint(
                        25, 45, 100
                    ),  # Tech-Savvy Remote Workers
                ]
            ),
            "Annual Income (k$)": np.concatenate(
                [
                    np.random.randint(70, 150, 50),  # Luxury Retirees
                    np.random.randint(40, 100, 50),  # Active Gamers
                    np.random.randint(10, 30, 50),  # Urban Sport Enthusiasts
                    np.random.randint(40, 80, 50),  # Gourmet Gardeners
                    np.random.randint(
                        50, 120, 100
                    ),  # Tech-Savvy Remote Workers
                ]
            ),
            "Spending Score (1-100)": np.concatenate(
                [
                    np.random.randint(40, 80, 50),  # Luxury Retirees
                    np.random.randint(60, 90, 50),  # Active Gamers
                    np.random.randint(20, 60, 50),  # Urban Sport Enthusiasts
                    np.random.randint(30, 70, 50),  # Gourmet Gardeners
                    np.random.randint(
                        50, 90, 100
                    ),  # Tech-Savvy Remote Workers
                ]
            ),
            "Region": np.concatenate(
                [
                    [3] * 10 + [2] * 20 + [1] * 20,  # Luxury Retirees
                    [3] * 20 + [2] * 20 + [1] * 10,  # Active Gamers
                    [3] * 50,  # Urban Sport Enthusiasts
                    [3] * 10 + [2] * 30 + [1] * 10,  # Gourmet Gardeners
                    [3] * 50 + [2] * 50,  # Tech-Savvy Remote Workers
                ]
            ),
            "Employment Status": np.concatenate(
                [
                    ["Retired"] * 50,  # Luxury Retirees
                    ["Employed"] * 50,  # Active Gamers
                    ["Student"] * 50,  # Urban Sport Enthusiasts
                    ["Employed"] * 50,  # Gourmet Gardeners
                    ["Employed"] * 100,  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Travel": np.concatenate(
                [
                    np.random.randint(60, 100, 50),  # Luxury Retirees
                    np.random.randint(1, 40, 50),  # Active Gamers
                    np.random.randint(1, 20, 50),  # Urban Sport Enthusiasts
                    np.random.randint(1, 30, 50),  # Gourmet Gardeners
                    np.random.randint(
                        30, 60, 100
                    ),  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Video Games": np.concatenate(
                [
                    np.random.randint(1, 20, 50),  # Luxury Retirees
                    np.random.randint(60, 100, 50),  # Active Gamers
                    np.random.randint(10, 40, 50),  # Urban Sport Enthusiasts
                    np.random.randint(10, 30, 50),  # Gourmet Gardeners
                    np.random.randint(
                        50, 90, 100
                    ),  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Sports": np.concatenate(
                [
                    np.random.randint(20, 60, 50),  # Luxury Retirees
                    np.random.randint(40, 80, 50),  # Active Gamers
                    np.random.randint(60, 100, 50),  # Urban Sport Enthusiasts
                    np.random.randint(30, 70, 50),  # Gourmet Gardeners
                    np.random.randint(
                        30, 70, 100
                    ),  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Dining": np.concatenate(
                [
                    np.random.randint(10, 50, 50),  # Luxury Retirees
                    np.random.randint(20, 60, 50),  # Active Gamers
                    np.random.randint(20, 50, 50),  # Urban Sport Enthusiasts
                    np.random.randint(60, 100, 50),  # Gourmet Gardeners
                    np.random.randint(
                        40, 70, 100
                    ),  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Gardening": np.concatenate(
                [
                    np.random.randint(20, 60, 50),  # Luxury Retirees
                    np.random.randint(10, 50, 50),  # Active Gamers
                    np.random.randint(10, 40, 50),  # Urban Sport Enthusiasts
                    np.random.randint(60, 100, 50),  # Gourmet Gardeners
                    np.random.randint(
                        20, 50, 100
                    ),  # Tech-Savvy Remote Workers
                ]
            ),
            "Interest in Music": np.concatenate(
                [
                    np.random.randint(30, 70, 50),  # Luxury Retirees
                    np.random.randint(20, 60, 50),  # Active Gamers
                    np.random.randint(20, 50, 50),  # Urban Sport Enthusiasts
                    np.random.randint(30, 70, 50),  # Gourmet Gardeners
                    np.random.randint(
                        50, 90, 100
                    ),  # Tech-Savvy Remote Workers
                ]
            ),
            "Household Income (k$)": np.concatenate(
                [
                    np.random.randint(70, 150, 50),  # Luxury Retirees
                    np.random.randint(40, 100, 50),  # Active Gamers
                    np.random.randint(10, 30, 50),  # Urban Sport Enthusiasts
                    np.random.randint(40, 80, 50),  # Gourmet Gardeners
                    np.random.randint(
                        50, 120, 100
                    ),  # Tech-Savvy Remote Workers
                ]
            ),
        }
    )
    return data


def main():
    data = load_data()
    # save data
    data.to_csv("clustering.csv", index=False)


if __name__ == "__main__":
    main()
