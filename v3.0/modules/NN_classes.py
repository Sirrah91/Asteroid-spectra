from modules._constants import _sep_in

def gimme_list_of_classes(grid_option: str) -> list[str]:
    if f"450{_sep_in}2450" in grid_option or f"650{_sep_in}2450" in grid_option or grid_option == "15":
        list_of_classes = [
            "A+",  # A + Sa
            "B",
            "C+",  # C + Cb + Cg
            "Ch+",  # Ch + Cgh
            "D",
            "K",
            "L",
            "Q",
            "S+",  # S + Sqw + Sw
            "Sr+",  # Sr + Srw + R
            "T",
            "V+",  # V + Vw
            "X+",  # X + Xc
            "Xe",
            "Xk",
        ]

    elif "HS-H-C" in grid_option:
        list_of_classes = [
            "B",
            "C+",  # C + Cb + Cg
        ]

    elif "HS-H-X" in grid_option:
        list_of_classes = [
            "X+",  # X + Xc + Xe
            "Xk",
        ]

    elif (f"500{_sep_in}900" in grid_option or
            f"650{_sep_in}1600" in grid_option or
            f"670{_sep_in}950" in grid_option or
            "ASPECT" in grid_option or
            "HS" in grid_option or
          grid_option == "9"):
        list_of_classes = [
            "A+",  # A + Sa
            "C+",  # C + Cb + Cg + B
            "Ch+",  # Ch + Cgh
            "D",
            "L",
            "Q",
            "S+",  # S + Sqw + Sr + Srw + Sw
            "V+",  # V + Vw
            "X+",  # X + Xc + Xe + Xk
        ]

    elif grid_option == "16":
        list_of_classes = [
            "A+",  # A + Sa
            "B",
            "C+",  # C + Cb
            "Cgh+",  # Cgh + Cg
            "Ch",
            "D",
            "K",
            "L",
            "Q",
            "S+",  # S + Sqw + Sw
            "Sr+",  # Sr + Srw + R
            "T",
            "V+",  # V + Vw
            "X+",  # X + Xc
            "Xe",
            "Xk",
        ]

    elif grid_option == "2":  # used for tests of space weathering for Ozgur
        list_of_classes = [
            "S",
            "Q",
        ]

    else:
        raise ValueError("Cannot connect model grid and list of classes.")

    list_of_classes_all = [
        "A",
        "B",
        "C",
        "Cb",
        "Cg",
        "Cgh",
        "Ch",
        "D",
        "K",
        "L",
        "O",
        "Q",
        "Qw",
        "R",
        "S",
        "Sa",
        "Sq",
        "Sq:",
        "Sqw",
        "Sr",
        "Srw",
        "Sv",
        "Svw",
        "Sw",
        "T",
        "U",
        "V",
        "Vw",
        "X",
        "Xc",
        "Xe",
        "Xk",
        "Xn",
    ]

    # must be here for tests of accuracy
    # list_of_classes = ["S", "Q"]

    return list_of_classes
