import numpy as np
from typing import Optional, Union, Type, List, Any
import math
from itertools import product
from enum import Enum
from copy import deepcopy

from numba.core.ir_utils import next_label


class CardMast(Enum):
    CLUBS = (1, "Clubs")
    DIAMONDS = (2, "Diamonds")
    HEARTS = (3, "Hearts")
    SPADES = (4, "Spades")

    def __init__(self, value, description):
        self._value_ = value
        self.description = description

class CardPower(Enum):
    TWO = (2, "Two")
    THREE = (3, "Three")
    FOUR = (4, "Four")
    FIVE = (5, "Five")
    SIX = (6, "Six")
    SEVEN = (7, "Seven")
    EIGHT = (8, "Eight")
    NINE = (9, "Nine")
    TEN = (10, "Ten")
    JACK = (11, "Jack")
    QUEEN = (12, "Queen")
    KING = (13, "King")
    ACE = (14, "Ace")

    def __init__(self, value, description):
        self._value_ = value
        self.description = description

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

class GameActionMomento:
    """
        Класс Memento — содержит действия игроков, которые можно сохранить.
        """

    def __init__(self, action: dict) -> None:
        self._state = deepcopy(action)  # Создаем копию состояния для сохранения

    def get_state(self) -> dict:
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

    def save(self, momento: Union[GameStateMomento, GameActionMomento]) -> None:
        """
        Сохраняем состояние игры в историю.
        """
        self.history.append(momento)

    def set_winner(self, winner: List[int]) -> None:
        self.winner = winner

    def restore(self, index: int) -> Union[GameStateMomento, GameActionMomento]:
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
        self.raw_bank = None

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
                self.raw_bank.append({"power": j, "type": i})

    def convert_cards(self, cards: np.ndarray) -> list[str]:
        names = []
        for i in cards:
            names.append(self.bank[np.argmax(self.logic_matrix @ i)])
        return names

    def redo(self):
        self.bank = []
        self.raw_bank = []
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

class FoolGame(Deck):
    def __init__(self, cards, num_types, num_cards, num_players, names: Optional[dict] = None, verbose: bool = False):
        """
        Инициализирует игру в дурака.

        Args:
            cards (int): Общее количество карт в колоде
            num_types (int): Количество мастей/типов карт
            num_cards (int): Количество карт, которые получает каждый игрок в начале
            num_players (int): Количество игроков
            names (dict, optional): Словарь с названиями карт. По умолчанию None
            verbose (bool): Флаг подробного вывода информации. По умолчанию False
        """
        super().__init__(cards, num_types, names)
        self.verbose = verbose
        self.num_players = num_players
        self.num_cards = num_cards

        self.players_banks = [] # [num_players, a, 6]
        self.players_info = [[] for i in range(num_players)] # [num_players, b, 6]
        self.table = [] # [c, 6]
        self.bita = [] # [d, 6]
        self.target = [] # [e, 6]
        self.finished_players = []

        self.busy_cards = []
        self.state_history = GameHistory()
        self.h = []

        self.round = 0
        self.role = 0
        self.last_fool = None
        self.cosir = None
        self.power_matrix = None

    def copy(self) -> 'FoolGame':
        new_game = self.__class__(
            self.cards,
            self.num_types,
            self.num_cards,
            self.num_players,
            deepcopy(self.names),  # Если names - mutable объект
            self.verbose
        )
        new_game.redo()
        new_game.players_banks = deepcopy(self.players_banks)
        new_game.players_info = deepcopy(self.players_info)
        new_game.table = deepcopy(self.table)
        new_game.bita = deepcopy(self.bita)
        new_game.target = deepcopy(self.target)
        new_game.finished_players = deepcopy(self.finished_players)
        new_game.busy_cards = deepcopy(self.busy_cards)
        new_game.h = deepcopy(self.h)
        new_game.role = self.role  # int - immutable, можно без deepcopy
        new_game.round = self.round  # int - immutable
        new_game.cosir = deepcopy(self.cosir)
        new_game.state_history = deepcopy(self.state_history)  # Если есть

        return new_game

    def redo(self):
        """
        Сбрасывает состояние игры для начала новой партии.
        Сохраняет информацию о последнем дураке для определения первого хода.
        """
        super().redo()
        self.players_banks = [[] for _ in range(self.num_players)]

        self.history = GameHistory()
        self.busy_cards = []
        if self.last_fool:
            self.round = (self.last_fool + 1) % self.num_players
            self.last_fool = None
        self.get_cards()

    def save_game(self):
        state_dict = {'players_banks': self.players_banks,
                      'players_info': self.players_info,
                      'table': self.table,
                      'bita': self.bita,
                      'target': self.target,
                      'finished_players': self.finished_players,
                      'busy_cards': self.busy_cards,
                      'h': self.h,
                      'role': self.role,
                      'round': self.round,
                      'cosir': self.cosir}
        self.state_history.save(GameActionMomento(state_dict))

    def retrieve(self, idx: int):
        state_dict = self.state_history.restore(idx).get_state()
        self.players_banks = state_dict['players_banks']
        self.players_info = state_dict['players_info']
        self.table = state_dict['table']
        self.bita = state_dict['bita']
        self.target = state_dict['target']
        self.finished_players = state_dict['finished_players']
        self.busy_cards = state_dict['busy_cards']
        self.h = state_dict['h']
        self.role = state_dict['role']
        self.round = state_dict['round']
        self.cosir = state_dict['cosir']

    def generate_power_matrix(self):
        """
        Генерирует матрицу силы карт, где каждая ячейка содержит количество карт,
        которые может побить данная карта с учетом козыря.
        """
        power_matrix = [[0 for _ in range(self.cards // self.num_types)] for _ in range(self.num_types)]
        for card in self.raw_bank:
            if self.cosir[1]['type'] == card['type']:
                func = lambda x: x['type'] != card['type'] or x['power'] < card['power']
            else:
                func = lambda x: x['power'] < card['power'] and x['type'] == card['type']

            power_matrix[card['type']][card['power']] = \
                len(tuple(filter(func, self.raw_bank)))
        self.power_matrix = np.array(power_matrix)

    def get_cards(self):
        """
        Раздает начальные карты всем игрокам и определяет козырь.
        Каждый игрок получает по num_cards карт.
        """
        for i in range(self.num_cards):
            for j in range(self.num_players):
                hash = np.random.randint(0, self.cards, 1)[0]
                while hash in self.busy_cards:
                    hash = np.random.randint(0, self.cards, 1)[0]
                self.busy_cards.append(hash)

                self.players_banks[j].append((hash, self.bank[hash], self.raw_bank[hash], self.logic_matrix[hash]))
                #self.players_banks[j].append(self.logic_matrix[hash])

        hash = np.random.randint(0, self.cards, 1)[0]
        while hash in self.busy_cards:
            hash = np.random.randint(0, self.cards, 1)[0]
        self.busy_cards.append(hash)

        self.cosir = (hash, self.bank[hash], self.raw_bank[hash], self.logic_matrix[hash])
        #self.cosir = [self.logic_matrix[hash]]

        self.generate_power_matrix()
        self.save_game()

    def get_cards_foreach(self, num_cards: int):
        """
        Добирает указанное количество карт каждому игроку до максимального количества.

        Args:
            num_cards (int): Количество карт, до которого нужно добрать каждому игроку
        """
        have_all = [False for _ in range(self.num_players)]
        have_given = 0
        while have_given < num_cards:
            for j in range(self.num_players):
                if len(self.players_banks[j]) >= self.num_cards:
                    have_all[j] = True
                    if all(have_all):
                        have_given = num_cards
                        break
                    continue
                if len(self.busy_cards) < self.cards:
                    hash = np.random.randint(0, self.cards, 1)[0]
                    while hash in self.busy_cards:
                        hash = np.random.randint(0, self.cards, 1)[0]
                    self.busy_cards.append(hash)

                    self.players_banks[j].append((self.bank[hash], self.logic_matrix[hash]))
                    have_given += 1
                else:
                    self.players_banks[j].append((self.cosir))
                    have_given = num_cards
                    break
                if have_given >= num_cards:
                    break
        self.save_game()

    def can_beat(self, card1, card2):
        """Проверяет, что карта 1 может покрыть карту 2"""
        if self.power_matrix[card1[2]['type']][card1[2]['power']] > self.power_matrix[card2[2]['type']][card2[2]['power']]:
            return True
        else:
            return False

    def can_beat_hash(self, card1_hash, card2_hash):
        """Проверяет, что карта с хэшем 1 покрывает карту с хэшем 2"""
        pow1 = self.raw_bank[card1_hash]['power']
        pow2 = self.raw_bank[card2_hash]['power']
        type1 = self.raw_bank[card1_hash]['type']
        type2 = self.raw_bank[card2_hash]['type']
        if self.power_matrix[type1][pow1] > self.power_matrix[type2][pow2]:
            return True
        else:
            return False

    def on_table(self, cards_hash) -> tuple[list, int]:
        """Принимает список хэшей карт. Проверяет атакующего, что карты данных рангов уже есть на столе.
        Возвращает список правильных карт и штраф > 0 за нарушение правил"""
        rew = 0
        good_hashes = []
        if len(cards_hash) == 0:
            return [], -10
        if self.table == [] and self.target == []:
            cards_pow = [self.raw_bank[hash]['power'] for hash in cards_hash]
            ok = cards_pow[0]
            for i, card_pow in enumerate(cards_pow):
                if card_pow == ok:
                    good_hashes.append(cards_hash[i])
                else:
                    rew -= 5
        else:
            table_pow = [card[2]['power'] for card in self.table]
            target_pow = [card[2]['power'] for card in self.target]
            cards_pow = [self.raw_bank[hash]['power'] for hash in cards_hash]
            for i, card_pow in enumerate(cards_pow):
                pows = set(target_pow + table_pow)
                if card_pow in pows:
                    good_hashes.append(cards_hash[i])
                else:
                    rew -= 5
        return good_hashes, rew

    def check_validity_att(self, actions) -> int:
        """Проверяет валидность действий атакующего, возвращает награду + штраф"""
        reward = 0
        old_round = self.round
        if (self.round % self.num_players) in self.finished_players:
            reward += 10
            self.h.append("done")
            self.round += 1
            #print('1 out')
            return reward
        if actions[0] == self.cards:
            if self.table == [] and self.target == []:
                reward -= 100
            self.add_information((self.round + 1) % self.num_players)
            self.h.append("done")
            self.round += 1
            #print('2 out')
        else:
            good_actions, rew = self.in_hand_hash(actions, self.round % self.num_players)
            reward += rew
            good_actions, rew = self.on_table(good_actions)
            reward += rew
            if len(good_actions) == 0:
                if len(self.h) > 0 and self.h[-1] == 'take':
                    self.add_information((self.round + 1) % self.num_players)
                    self.round += 1
                self.h.append("done")
                self.round += 1
                self.bita += self.table
                self.table = []
                #print('3 out')
            else:

                self.target += [(act_hash, self.bank[act_hash], self.raw_bank[act_hash], self.logic_matrix[act_hash]) for act_hash in good_actions]
                self.delete_information(self.round % self.num_players, self.target)
                if len(self.h) > 0 and self.h[-1] == "take":
                    self.h.append("done")
                    self.round += 1
                    self.add_information(self.round % self.num_players)
                    self.round += 1
                    #print('4 out')
                else:
                    self.h.append("attacked")
                    self.round += 1
                    self.role = (self.role + 1) % 2
                    #print('5 out')
        if len(self.players_banks[old_round % self.num_players]) == 0 and (old_round % self.num_players) not in self.finished_players:
            reward += 100
            self.finished_players.append(old_round % self.num_players)
            if self.round == old_round:
                self.round += 1
        return reward

    def check_att_one_act(self, action: int) -> int:
        """Проверяет валидность действия атакующего, возвращает награду + штраф"""
        reward = 0
        old_round = self.round
        # Проверка, что игрок закончил ход
        if (self.round % self.num_players) in self.finished_players:
            reward = 0
            self.h.append("done")
            self.round += 1
            #print('1 out')
            return reward

        # Случай завершения хода
        if action == self.cards:
            if self.table == [] and self.target == []:
                reward = 0
            else:
                self.role = (self.role + 1) % 2
            self.add_information((self.round + 1) % self.num_players)
            self.h.append("done")
            self.round += 1
            #print('2 out')
        # Случай продолжения атаки
        else:
            self.target += [(action, self.bank[action], self.raw_bank[action], self.logic_matrix[action])]
            self.delete_information(self.round % self.num_players, self.target)
            self.h.append("attacked")
            #print('5 out')
        # Проверка, закончил ли игрок колоду
        if len(self.players_banks[old_round % self.num_players]) == 0 and (old_round % self.num_players) not in self.finished_players:
            reward += 100
            self.finished_players.append(old_round % self.num_players)
            if self.round == old_round:
                self.round += 1
            #if self.target != []:
            #    self.role = (1 + self.role) % 2

        return reward

    def check_validity_def(self, actions) -> int:
        """проверяет валидность действий защиты, возвращает награду + штраф"""
        reward = 0
        if actions[0] == self.cards:
            #print("1 out d")
            if len(self.target) == 0:
                reward += 10
                self.bita += self.table
                self.table = []
                self.role = (self.role + 1) % 2
            else:
                self.h.append("take")
                self.round -= 1
                self.role = (self.role + 1) % 2
        else:
            good_actions, rew = self.in_hand_hash(actions, self.round % self.num_players)
            reward += rew

            cards_hash = []
            cards = []
            unbeaten_cards = []
            beaten_cards = []
            covered = True
            for target_card in self.target:
                can_beat = False
                for i, card_hash in enumerate(good_actions):
                    if card_hash not in cards_hash:
                        card = (card_hash, self.bank[card_hash], self.raw_bank[card_hash], self.logic_matrix[card_hash])
                        if self.can_beat(card, target_card):
                            can_beat = True
                            beaten_cards.append(target_card)
                            cards_hash += [card_hash]
                            cards += [card]
                            break

                if not can_beat:
                    unbeaten_cards.append(target_card)
                    covered = False

            if covered or len(self.players_banks[self.round % self.num_players]) == len(cards):
                self.h.append("covered")
                self.table += cards + beaten_cards
                self.target = []
                self.delete_information(self.round % self.num_players, cards)
                self.role = (self.role + 1) % 2

                if len(self.players_banks[self.round % self.num_players]) == 0:
                    self.finished_players.append(self.round % self.num_players)
                    reward += 100
                    #print('4 out')

                    self.bita += self.table
                    self.players_banks[(self.round - 1) % self.num_players] += unbeaten_cards
                    self.players_info[(self.round - 1) % self.num_players] += unbeaten_cards
                    self.table = []
                    return reward
                reward += 20
                self.round -= 1

                #print("2 out d")
            else:
                self.h.append("take")
                self.round -= 1
                self.role = (self.role + 1) % 2
                #print("3 out d")

        return reward

    def check_def_one_act(self, action: int) -> int:
        """проверяет валидность действия защиты, возвращает награду + штраф"""
        reward = 0
        if action == self.cards:
            #print("1 out d")
            if len(self.target) == 0:
                reward += 10
                self.bita += self.table
                self.table = []
                self.role = (self.role + 1) % 2
            else:
                self.h.append("take")
                self.round -= 1
                self.role = (self.role + 1) % 2
        else:
            cards = []
            new_tgt = []
            covered = False
            for target_card in self.target:
                card = (action, self.bank[action], self.raw_bank[action], self.logic_matrix[action])
                if self.can_beat(card, target_card) and not covered:
                    self.table += [target_card] + [card]
                    cards += [card]
                    covered = True
                else:
                    new_tgt.append(target_card)
            self.target = new_tgt
            self.delete_information(self.round % self.num_players, cards)

            if len(self.target) == 0 or len(self.players_banks[self.round % self.num_players]) == 0:
                self.h.append("covered")

                self.role = (self.role + 1) % 2

                if len(self.players_banks[self.round % self.num_players]) == 0:
                    self.finished_players.append(self.round % self.num_players)
                    reward += 100
                    #print('4 out')

                    self.bita += self.table
                    self.players_banks[(self.round - 1) % self.num_players] += self.target
                    self.players_info[(self.round - 1) % self.num_players] += self.target
                    self.target = []
                    self.table = []
                    return reward
                reward += 20
                self.round += 1
                #print("2 out d")

        return reward

    def in_hand_hash(self, card_hashes, player_num):
        """Проверяет, какие карты есть в руке, и возвращает имеющиеся карты и штраф > 0 за не имеющиеся"""
        reward = 0
        good_hashes = []
        for i in self.players_banks[player_num]:
            have = False
            for card_hash in card_hashes:
                if card_hash == i[0]:
                    good_hashes.append(card_hash)
                    have = True
                    continue
            if not have:
                reward -= 10

        return good_hashes, reward

    def add_information(self, player_num):
        """добавляет игроку player_num карты из target и table
        и обнуляет target и table"""
        self.players_info[player_num] += self.target + self.table
        self.players_banks[player_num] += self.target + self.table
        self.target = []
        self.table = []

    def delete_information(self, player_num, cards):
        """Удаляет карты из информации игроков и из карт игрока"""
        # Создаем списки индексов для удаления
        to_delete_info = []
        to_delete_banks = []

        # Находим индексы для удаления в players_info
        for i, card_info in enumerate(self.players_info[player_num]):
            for card in cards:
                if card[0] == card_info[0]:
                    to_delete_info.append(i)
                    break  # одна карта может совпасть только один раз

        # Находим индексы для удаления в players_banks
        for i, card_bank in enumerate(self.players_banks[player_num]):
            for card in cards:
                if card[0] == card_bank[0]:
                    to_delete_banks.append(i)
                    break  # одна карта может совпасть только один раз

        # Удаляем элементы в обратном порядке, чтобы индексы не сдвигались
        for i in sorted(to_delete_info, reverse=True):
            del self.players_info[player_num][i]

        for i in sorted(to_delete_banks, reverse=True):
            del self.players_banks[player_num][i]

    def step_many(self, actions: list[int], save=True) -> tuple[float, bool]:
        actions = list(dict.fromkeys(actions))
        if len(self.finished_players) == self.num_players - 1:
            reward = 0
            return reward, False
        reward = 0
        #print("round before:", self.round)
        while self.round % self.num_players in self.finished_players:
            self.round += 1
        #print("player:", self.round % self.num_players)
        #print("player cards:", [card[0] for card in self.players_banks[self.round % self.num_players]])
        if self.role == 0: # Attacker
            reward = self.check_validity_att(actions)
        elif self.role == 1:
            reward = self.check_validity_def(actions)
        #print("round after:", self.round)
        if save:
            self.save_game()
        return reward, True

    def step(self, action: int, save: bool = True) -> tuple[float, bool, bool]:
        """returns:
        reward, match has ended, player has ended"""
        reward = 0
        player = self.round % self.num_players
        if len(self.finished_players) == self.num_players - 1:
            #TODO reward
            reward = 1
            if player in self.finished_players:
                terminal = True
            else:
                terminal = False
            return reward, False, not terminal

        #print("round before:", self.round)
        while self.round % self.num_players in self.finished_players:
            self.round += 1
        #print("player:", self.round % self.num_players)
        #print("player cards:", [card[0] for card in self.players_banks[self.round % self.num_players]])
        if self.role == 0: # Attacker
            reward = self.check_att_one_act(action)
        elif self.role == 1:
            reward = self.check_def_one_act(action)
        #print("round after:", self.round)
        if save:
            self.save_game()
        if player in self.finished_players:
            terminal = True
        else:
            terminal = False
        finished_players = []
        for i, player in enumerate(self.players_banks):
            if len(player) == 0:
                finished_players.append(i)
        self.finished_players = finished_players
        return reward, True, not terminal

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

class GameState:
    def __init__(self, data, current_player, is_terminal=False):
        self.data = data  # произвольное состояние (например, массив карт)
        self.current_player = current_player  # int: 0, 1, 2
        self.is_terminal = is_terminal

    def get_legal_actions(self):
        """Вернуть список допустимых действий в текущем состоянии"""
        raise NotImplementedError

    def apply_action(self, action):
        """Применить действие и вернуть новое состояние"""
        raise NotImplementedError

    def evaluate(self):
        """Возвращает оценку выигрыша для всех игроков"""
        raise NotImplementedError

class GameTreeNode:
    def __init__(self, state: GameState, parent=None):
        self.state = state                  # GameState
        self.parent: GameTreeNode = parent                # родительский узел
        self.action = action                # действие, которое привело к этому узлу
        self.children = []                  # список дочерних узлов

    def is_terminal(self):
        return self.state.is_terminal

    def expand(self):
        """Создаёт дочерние узлы для всех допустимых действий"""
        actions = self.state.get_legal_actions()
        for action in actions:
            next_state = self.state.apply_action(action)
            child_node = GameTreeNode(next_state, parent=self, action=action)
            self.children.append(child_node)

    def match_previous_action(self) -> tuple[GameState, int]:
        parent = self.parent
        while parent.state.current_player != self.state.current_player:
            action = parent.action
            parent = parent.parent
        if parent.parent.state.current_player == self.state.current_player:
            return self.state, parent.parent.state['q_val'], parent.state['rew']

    def traverse(self, depth=0):
        """Рекурсивный обход дерева (для отладки)"""
        print("  " * depth + f"Player {self.state.current_player}, Action: {self.action}")
        for child in self.children:
            child.traverse(depth + 1)


if __name__ == '__main__':
    # names = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10', 9: "jack", 10: "queen", 11: "king", 12: "ace"} #, 13: "d_ace", 14: "t_ace"}
    # game = PokerGame(52, 4, 2, 3, 2, 3, names=names)
    #
    # game.start_new_game()
    #
    # cards = np.array([[0, 1, 0, 0, 1, 0],
    #         [0, 1, 1, 0, 0, 1],])
    # print(game.convert_cards(cards))

    names = {0: '6', 1: '7', 2: '8', 3: '9', 4: '10', 5: "jack", 6: "queen",
             7: "king", 8: "ace"}
    game = FoolGame(36, 4, 6, 3, names=names)
    game.redo()
    # for player in game.players_banks:
    #     print(len(player))
    # print("__________")
    # print("__________")
    # print(game.power_matrix)
    # print(game.raw_bank)
    # print(game.cosir)
    # print(game.busy_cards)
    # print(game.players_banks)
    # print(game.logic_matrix)
    fake_actions = np.random.randint(0, 37, size=100)
    print(fake_actions)

    for i, action in enumerate(fake_actions):
        print("action", action)
        # print("role", game.role)
        print("turn =", i)
        print("finished -", game.finished_players)
        print('role:', game.role)
        rew, next, player_finished = game.step(action, False)

        # print(rew)
        # print(next)
        # print("::::")
        # print("finished:", game.finished_players)
        for player in game.players_banks:
            print(len(player))
        # print(len(game.bita))
        # print(len(game.table))
        # print(len(game.target))
        # print("__________")
        # print()
        if not next:
            break