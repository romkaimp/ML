import numpy as np
from typing import Optional, Union, Type, List, Any
import math
from itertools import product
from enum import Enum
from copy import deepcopy

class GameStateMomento:
    """
    Класс Memento — содержит состояние игры, которое можно сохранить.
    """
    def __init__(self, state: List[List[np.ndarray]]) -> None:
        self._state = deepcopy(state)  # Создаем копию состояния для сохранения

    def get_state(self) -> List[List[np.ndarray]]:
        """
        Метод для получения сохраненного состояния.
        """
        return deepcopy(self._state)

    def __repr__(self):
        return str(self._state)

class GameHistory:
    """
    Хранитель (Caretaker) — отвечает за хранение всех сохраненных состояний.
    """
    def __init__(self) -> None:
        self.history: List[GameStateMomento] = []
        self.winner: List[int] = []

    def save(self, momento: GameStateMomento) -> None:
        """
        Сохраняем состояние игры в историю.
        """
        self.history.append(momento)

    def set_winner(self, winner: List[int]) -> None:
        self.winner = winner

    def restore(self, index: int) -> GameStateMomento:
        """
        Получаем сохранённое состояние по индексу.
        """
        return self.history[index]

class Embeddings:
    def __init__(self, cards, num_types):
        self.cards = cards
        self.num_types = num_types

        assert cards % num_types == 0, "cards must be divisible by num_types"
        self.card_dim = math.ceil(math.log2(cards / num_types + 1))  # 0 is reserved
        self.type_dim = math.ceil(math.log2(num_types))
        self.logic_matrix = self.generate_embeddings()
        self.dim = self.type_dim + self.card_dim

    def generate_embeddings(self):
        suits = np.array([[int(b) for b in format(i, f'0{self.type_dim}b')] for i in range(self.num_types)])
        ranks = np.array(
            [[int(b) for b in format(i + 1, f'0{self.card_dim}b')] for i in range(self.cards // self.num_types)])
        deck = np.array([[*suit, *rank] for suit in suits for rank in ranks])
        return deck

class Deck:
    '''Базовый класс с колодами для разных игр'''
    def __init__(self, cards, num_types, names: Optional[dict] = None):
        self.bank = None #[]

        assert cards % num_types == 0, "cards must be divisible by num_types"
        self.cards = cards
        self.num_types = num_types

        assert len(names) == cards // num_types, "names size must equal cards/num_types"
        self.names = names

        self.embeddings = Embeddings(cards, num_types)
        self.logic_matrix = self.embeddings.logic_matrix

    def generate_cards(self):
        for i in range(self.num_types):
            for j in range(self.cards // self.num_types):

                if self.names:
                    self.bank.append({"power": self.names[j], "type": i})
                else:
                    self.bank.append({"power": j, "type": i})

    def convert_cards(self, cards: np.ndarray) -> list:
        names = []
        for i in cards:
            names.append(self.bank[np.argmax(self.logic_matrix @ i)])
        return names

    def redo(self):
        self.bank = []
        self.generate_cards()
        #self.logic_matrix = self.embeddings.logic_matrix

class Poker(Deck):
    '''В этом классе реализуются особенности покерной колоды'''
    def __init__(self, cards, num_types, num_cards, flop_size, names: Optional[dict] = None,):
        super().__init__(cards, num_types, names)
        self.flop_size = flop_size
        self.num_cards = num_cards

class PokerHand(Enum):
    HIGH_CARD = (1, "High Card")
    ONE_PAIR = (2, "One Pair")
    TWO_PAIR = (3, "Two Pair")
    THREE_OF_A_KIND = (4, "Three of a Kind")
    STRAIGHT = (5, "Straight")
    FLUSH = (6, "Flush")
    FULL_HOUSE = (7, "Full House")
    FOUR_OF_A_KIND = (8, "Four of a Kind")
    STRAIGHT_FLUSH = (9, "Straight Flush")
    ROYAL_FLUSH = (10, "Royal Flush")

    def __init__(self, value, description):
        self._value_ = value
        self.description = description

class PokerHandEvaluator:
    def __init__(self, hand, embeddings: Embeddings, verbose: bool = False):
        """
        Принимает список карт (по 6 бит на каждую).
        """
        self.verbose = verbose
        self.hand = np.array(hand)

        self.num_types = embeddings.num_types
        self.cards = embeddings.cards

        self.suit_mask = np.zeros(self.num_types, dtype=int)  # Маска для мастей
        self.rank_mask = np.zeros(self.cards//self.num_types + 1, dtype=int)  # Маска для значений

        self.card_dim, self.type_dim = embeddings.card_dim, embeddings.type_dim

        self.max_rank = 0

    def parse_hand(self):
        """Разбирает карты, создавая битовые маски мастей и значений"""
        max_rank = 0
        for hand in self.hand:
            bin_num = BinaryOps.list_to_bin(hand)

            suit = (bin_num >> self.card_dim) & BinaryOps.get_binary_ones(self.type_dim) # Первые type_dim бита (масть)
            rank = bin_num & BinaryOps.get_binary_ones(self.card_dim)  # Последние card_dim бита (значение)
            self.suit_mask[suit] += 1
            self.rank_mask[rank] += 1

            max_rank = max(max_rank, rank)

        self.max_rank = max_rank

    def is_flush(self):
        """Проверяет, есть ли флеш (5+ карт одной масти)"""
        return np.any(self.suit_mask >= 5)

    def is_straight(self):
        """Проверяет, есть ли стрит (5 подряд идущих значений)"""
        bit_pattern = 0
        for i in range(len(self.rank_mask)):
            if self.rank_mask[i] > 0:
                bit_pattern |= (1 << i)

        if self.verbose:
            print(bin(bit_pattern))
        # Проверяем наличие 5 подряд идущих единиц
        for i in range(self.cards // self.num_types - 4):  # -5 + 1, так как 1 лишняя карта (пустота)
            if (bit_pattern >> i) & 0b11111 == 0b11111:
                return True

        # Проверка особого случая: стрит A-2-3-4-5 (Ace = 13, 2 = 0)
        if ((bit_pattern & (BinaryOps.get_binary_ones(4) << 1 | int(bin(2 ** (self.cards // self.num_types))[2:], 2))) ==
                BinaryOps.get_binary_ones(4) << 1 | int(bin(2 ** (self.cards // self.num_types))[2:], 2)):
            return True

        return False

    def get_hand_rank(self):
        """Определяет лучшую комбинацию"""
        self.parse_hand()

        if self.is_flush() and self.is_straight() and self.rank_mask[-1] != 0 and self.rank_mask[-2] != 0:
            return PokerHand.ROYAL_FLUSH

        # 1. Роял-флеш и стрит-флеш
        if self.is_flush() and self.is_straight():
            return PokerHand.STRAIGHT_FLUSH

        # 2. Каре
        if 4 in self.rank_mask:
            return PokerHand.FOUR_OF_A_KIND

        # 3. Фулл-хаус
        if 3 in self.rank_mask and 2 in self.rank_mask:
            return PokerHand.FULL_HOUSE

        # 4. Флеш
        if self.is_flush():
            return PokerHand.FLUSH

        # 5. Стрит
        if self.is_straight():
            return PokerHand.STRAIGHT

        # 6. Сет (тройка)
        if 3 in self.rank_mask:
            return PokerHand.THREE_OF_A_KIND

        # 7. Две пары
        if np.count_nonzero(self.rank_mask == 2) == 2:
            return PokerHand.TWO_PAIR

        # 8. Одна пара
        if 2 in self.rank_mask:
            return PokerHand.ONE_PAIR

        # 9. Старшая карта
        return PokerHand.HIGH_CARD

class PokerGame(Poker):
    def __init__(self, cards, num_types, num_cards, flop_size, num_players, rounds, names: Optional[dict] = None, verbose: bool = False):
        super().__init__(cards, num_types, num_cards, flop_size, names,)
        self.verbose = verbose
        self.num_players = num_players

        self.players_banks = None #[[] for _ in range(self.num_players)]
        self.table = None #[]

        self.round = 0
        self.rounds = rounds
        self.history = None #GameHistory()
        self.busy_cards = None #[]

    def save(self) -> None:
        all_players_cards = []

        for i in range(self.num_players):
            player_cards = self.players_banks[i] + self.table
            player_cards = list(map(lambda x: x[1], player_cards))
            all_players_cards.append(player_cards)

        state = GameStateMomento(all_players_cards)
        self.history.save(state)

    def get_cards(self):
        for i in range(self.num_cards):
            for j in range(self.num_players):
                hash = np.random.randint(0, self.cards, 1)[0]
                while hash in self.busy_cards:
                    hash = np.random.randint(0, self.cards, 1)[0]
                self.busy_cards.append(hash)

                self.players_banks[j].append((self.bank[hash], self.logic_matrix[hash]))

    def new_card_on_table(self):
        hash = np.random.randint(0, self.cards, 1)[0]
        while hash in self.busy_cards:
            hash = np.random.randint(0, self.cards, 1)[0]
        self.busy_cards.append(hash)
        self.table.append((self.bank[hash], self.logic_matrix[hash]))

    def flop(self):
        for i in range(self.flop_size):
            hash = np.random.randint(0, self.cards, 1)[0]
            while hash in self.busy_cards:
                hash = np.random.randint(0, self.cards, 1)[0]
            self.busy_cards.append(hash)
            self.table.append((self.bank[hash], self.logic_matrix[hash]))

    def redo(self):
        super().redo()
        self.players_banks = [[] for _ in range(self.num_players)]
        self.table = []
        self.history = GameHistory()
        self.busy_cards = []

    def start_new_game(self):
        self.redo()

        self.get_cards()
        self.flop()
        #TODO get_bids()
        self.save()
        for i in range(self.rounds - 1):
            self.new_card_on_table()
            #TODO get_bids()
            self.save()
        winners = self.check_winner()
        self.history.set_winner(winners)

    def get_history(self) -> dict[str, Any]:
        return {"history": self.history.history, "winner": self.history.winner}

    def get_player_value(self, i) -> tuple[PokerHand, int]:
        '''Возвращает руку игрока и максимальную карту'''
        player_cards = self.players_banks[i] + self.table
        player_cards = list(map(lambda x: x[1], player_cards))

        hand = PokerHandEvaluator(player_cards, self.embeddings)
        hand_value, rank_value = hand.get_hand_rank(), hand.max_rank
        return hand_value, rank_value

    def check_winner(self) -> List[int]:
        best_hand_value = PokerHand.HIGH_CARD
        best_rank_value = 0
        winners = []

        for i in range(self.num_players):
            hand_value, rank_value = self.get_player_value(i)

            if hand_value.value > best_hand_value.value:
                best_hand_value = hand_value
                best_rank_value = rank_value
                winners = [i]  # Новый лучший игрок, обнуляем список
            elif hand_value.value == best_hand_value.value:
                if rank_value > best_rank_value:
                    best_rank_value = rank_value
                    winners = [i]  # Новый лидер по старшей карте
                elif rank_value == best_rank_value:
                    winners.append(i)  # Добавляем в список победителей

        return winners

class BinaryOps:
    @staticmethod
    def list_to_bin(array) -> int:
        binary_number = 0
        for bit in array:
            binary_number = (binary_number << 1) | bit

        return int(bin(binary_number)[2:], 2)

    @staticmethod
    def get_binary_ones(n) -> int:
        binary_number = (1 << n) - 1
        return int(bin(binary_number)[2:], 2)



if __name__ == '__main__':
    names = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10', 9: "jack", 10: "queen", 11: "king", 12: "ace"} #, 13: "d_ace", 14: "t_ace"}
    game = PokerGame(52, 4, 2, 3, 2, 3, names=names)

    game.start_new_game()

    cards = [[0, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
             [0, 1, 0, 0, 1, 0]]
    print(game.convert_cards(cards))

    hist = game.get_history()
    hand_eval = PokerHandEvaluator(cards, game.embeddings)
    print(hand_eval.get_hand_rank())
    print(hist)



