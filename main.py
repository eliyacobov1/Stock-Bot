EMA_SMOOTHING, INITIAL_RSI_WINDOW_SIZE, INITIAL_EMA_WINDOW_SIZE = 2, 14, 200


class StockBot:
    def __init__(self, rsi_win_size, ema_win_size):
        self.ema = None
        self.gain_avg = None
        self.loss_avg = None
        self.rsi_window_size, self.ema_window_size = rsi_win_size, ema_win_size

    @staticmethod
    def calc_shift_percentage(val1, val2):
        """
        Calculates the relative percentage between val1 and val2
        """
        return abs((val1 / val2) - 1) * 100

    @staticmethod
    def get_next_avg(curr_avg, val, window_size):
        """
        This function returns the weighted average of 'curr_avg' along with the new value 'val'
         according to the given window size
        """
        return (curr_avg * (window_size - 1) + val) / window_size

    def update_ema(self, val):
        """
        Calculates and updates the next EMA value according to the given new stock value
        """
        coefficient = EMA_SMOOTHING / (1 + self.ema_window_size)
        self.ema = val * coefficient + self.ema * (1 - coefficient)
        return self.ema

    def get_initial_rsi(self, initial_stock_values):
        """
        Calculates the initial RSI value according to a given starting period window of the stock.
        :return the initial RSI value, gain average and loss average of the stock
        """
        losses, gains, period_size = [], [], len(initial_stock_values)
        for open_price, close_price in initial_stock_values:
            diff_percentage = self.calc_shift_percentage(close_price, open_price)
            if close_price >= open_price:
                gains.append(diff_percentage)
            else:
                losses.append(diff_percentage)
        self.gain_avg, self.loss_avg = sum(gains) / len(gains), sum(losses) / len(losses)
        rsi = 100 - (100 / (1 + (self.gain_avg / self.loss_avg)))
        return rsi

    def get_rsi(self):
        """
        Calculates the RSI value according to the current gain and loss averages
        """
        return 100 - (100 / (1 + (self.gain_avg / self.loss_avg)))

    def update_rsi(self, gain, loss, window_size):
        """
        Calculates and updates the next RSI value according to the given new gain and loss
        """
        self.gain_avg = self.get_next_avg(self.gain_avg, gain, window_size)
        self.loss_avg = self.get_next_avg(self.loss_avg, loss, window_size)
        return self.get_rsi()

    def main(self):
        pass


if __name__ == '__main__':
    sb = StockBot(INITIAL_RSI_WINDOW_SIZE, INITIAL_EMA_WINDOW_SIZE)
    sb.main()

