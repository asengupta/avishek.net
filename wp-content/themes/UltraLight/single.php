<?php
/*
Template Name: Single
*/
?>

<?php get_header(); ?>

		<!-- Maincontent start -->
		<div id="maincontent">
			<h2>Posted in <?php the_category(' '); ?></h2>

<?php while (have_posts()) : the_post(); ?>
			<div class="singlepost">
				<h3><?php the_title(); ?></h3>
				<div class="date"><?php the_date(); ?> - <?php the_time(); ?></div>
				<?php the_content(); ?>
				<div class="postmeta">
					<span class="left"><?php if (function_exists('the_tags')) the_tags('Tags: '); ?></span>
				</div>
			</div>
			<?php comments_template();?>
<?php endwhile; ?>

			<div id="pagination" class="clear">
				<span class="left"><?php previous_post_link(); ?></span>
				<span class="right"><?php next_post_link(); ?></span>
			</div>
			<div id="navigation" class="clear">
				<span class="left"><a href="javascript:history.back(-1)" class="left">&laquo; Back</a></span>
				<?php edit_post_link(' Edit post', '<span class="right">', '</span>'); ?>
			</div>

		</div>
		<!-- Maincontent end -->

<?php get_sidebar(); ?>

<?php get_footer(); ?>
