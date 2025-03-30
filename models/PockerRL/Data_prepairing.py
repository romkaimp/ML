from Environment import PokerGame




if __name__ == '__main__':
    names = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10', 9: "jack", 10: "queen", 11: "king", 12: "ace"} #, 13: "d_ace", 14: "t_ace"}
    game = PokerGame(52, 4, 2, 3, 2, 3, names=names)

    game.start_new_game()
    history = game.get_history()
    print(history)