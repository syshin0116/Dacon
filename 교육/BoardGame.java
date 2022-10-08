// in BoardGame1.java


// package BoardGame;
public class BoardGame {
   public static void main(String[] args) {
      Board bd = new Board(5); // 5�� Place�� ��ȣ������ ������ Board ����
      Place p = bd.getStartPlace();
      System.out.println("Start Place = " + p.toString()); // p.toString() ��� p�� �־ ��
      System.out.println(p.getNext()); // p�� Next Place�� ��� or p.getNext().toString()
      // bd.print(); // board ��ü�� ���
   }

   static class Board {
      Place startPlace;
      Place finishPlace;

      int N;
      // ĭ�� ����

      public Board(int N) {
         Place[] p = new Place[N];
         // p[0] = new Place(0);         
         for (int i=0; i<p.length; i++){
            p[N] = new Place(i);
         }
         this.startPlace = p[0];
         this.finishPlace = p[N-1];
         // this.place = N;

         // System.out.println("N = " + place);

      }

      public Place getStartPlace() {
         return this.startPlace;
      }
      public void print(){
         System.out.println("N=" + N);

      }
   }

   static class Place {
      Place NextPlace;
      int No;

      public Place(int n){
         this.No = n;
      }
      public void setNext(Place p){
         p = this.NextPlace;
      }
      public Place getNext(){
         return this.NextPlace;
      }
      public int getNo(){
         return this.No;
      }
      public String toString(){
         String result;
         if (this.getNext().getNo() == NextPlace.No){
            result = "Current No. " + String.valueOf(this.getNo()) + "- null";
         }else{
         result = "Current No. " + String.valueOf(this.getNo()) + " - Next No. " + String.valueOf(this.getNo()+1);
         }
         return result;
      }
   }
}
